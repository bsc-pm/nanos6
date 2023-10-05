/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#include <iostream>

#include "CUDAAccelerator.hpp"
#include "CUDADirectoryDevice.hpp"
#include "scheduling/Scheduler.hpp"

#include <api/nanos6/events.h>

void CUDADirectoryDevice::memcpyFrom(DirectoryPage *page, DirectoryDevice *src, size_t size, void *srcAddress, void *dstAddress)
{
	assert(canCopyFrom(src));

	_accelerator->setActiveDevice();

	OngoingCopy copy;
	copy.destinationDevice = this->getId();
	CUDAFunctions::createEvent(copy.event);
	if (src->getType() == oss_device_host) {
		CUDAFunctions::copyMemoryAsync(dstAddress, srcAddress, size, cudaMemcpyHostToDevice, _directoryStream);
	} else {
		int srcDevice = ((CUDADirectoryDevice *) src)->_accelerator->getDeviceHandler();
		int dstDevice = _accelerator->getDeviceHandler();
		// P2P
		CUDAFunctions::copyMemoryP2PAsync(dstAddress, dstDevice, srcAddress, srcDevice, size, _directoryStream);
	}
	CUDAFunctions::recordEvent(copy.event, _directoryStream);
	page->_copyHandlers[this->getId()] = (void *) copy.event;
	copy.page = page;

	_lock.lock();
	_pendingEvents.push_back(copy);
	_lock.unlock();
}

//TODO merge both functions
void CUDADirectoryDevice::memcpyFromImplicit(DirectoryPage *page, DirectoryDevice *src, size_t size, void *srcAddress, void *dstAddress, Task *task)
{
	assert(canCopyFrom(src));

	cudaStream_t stream = task->getDeviceEnvironment().cuda.stream;
	OngoingCopy copy;
	copy.destinationDevice = this->getId();
	CUDAFunctions::createEvent(copy.event);
	if (src->getType() == oss_device_host) {
		CUDAFunctions::copyMemoryAsync(dstAddress, srcAddress, size, cudaMemcpyHostToDevice, stream);
	} else {
		int srcDevice = ((CUDADirectoryDevice *) src)->_accelerator->getDeviceHandler();
		int dstDevice = _accelerator->getDeviceHandler();
		// P2P
		CUDAFunctions::copyMemoryP2PAsync(dstAddress, dstDevice, srcAddress, srcDevice, size, stream);
	}
	CUDAFunctions::recordEvent(copy.event, stream);
	copy.page = page;
	page->_copyHandlers[this->getId()] = (void *) copy.event;

	_lock.lock();
	_pendingEvents.push_back(copy);
	_lock.unlock();
}

void CUDADirectoryDevice::synchronizeOngoing(DirectoryPage *page, Task *task)
{
	cudaEvent_t event = (cudaEvent_t) page->_copyHandlers[this->getId()];
	cudaStream_t stream = task->getDeviceEnvironment().cuda.stream;

	CUDAFunctions::waitForEvent(event, stream);
}

void CUDADirectoryDevice::memcpy(DirectoryPage *page, DirectoryDevice *dst, size_t size, void *srcAddress, void *dstAddress)
{
	assert(canCopyTo(dst));

	_accelerator->setActiveDevice();
	OngoingCopy copy;
	CUDAFunctions::createEvent(copy.event);
	copy.destinationDevice = dst->getId();

	assert(dst->getType() == oss_device_host);
	CUDAFunctions::copyMemoryAsync(dstAddress, srcAddress, size, cudaMemcpyDeviceToHost, _directoryStream);
	CUDAFunctions::recordEvent(copy.event, _directoryStream);
	copy.page = page;

	// TODO: This could be a lock-free MPSC queue.
	_lock.lock();
	_pendingEvents.push_back(copy);
	_lock.unlock();
}

static inline DirectoryPageState finishTransition(DirectoryPageState oldState)
{
	switch (oldState) {
		case StateTransitionShared:
			return StateShared;
		case StateTransitionModified:
			return StateModified;
		case StateTransitionExclusive:
			return StateExclusive;
		default:
			FatalErrorHandler::fail("Invalid State Transition");
			// Otherwise GCC complains
			return StateExclusive;
	}
}

void CUDADirectoryDevice::processEvents()
{
	bool finished = true;

	_lock.lock();
	_pendingEventsCopy.insert(_pendingEventsCopy.end(), _pendingEvents.begin(), _pendingEvents.end());
	_pendingEvents.clear();
	_lock.unlock();

	if (!_pendingEventsCopy.empty())
		_accelerator->setActiveDevice();

	while (finished && !_pendingEventsCopy.empty()) {
		OngoingCopy &copy = _pendingEventsCopy.front();
		finished = CUDAFunctions::isEventFinished(copy.event);

		if (finished) {
			// Notify page
			DirectoryPage *page = copy.page;
			CUDAFunctions::destroyEvent(copy.event);

			page->lock();

			page->_states[copy.destinationDevice] = finishTransition(page->_states[copy.destinationDevice]);

			for (Task *t : page->_pendingNotifications[copy.destinationDevice]) {
				if (t->decreasePredecessors())
					Scheduler::addReadyTask(t, nullptr, SIBLING_TASK_HINT);
			}

			page->_pendingNotifications[copy.destinationDevice].clear();
			page->unlock();

			_pendingEventsCopy.pop_front();
		}
	}
}
