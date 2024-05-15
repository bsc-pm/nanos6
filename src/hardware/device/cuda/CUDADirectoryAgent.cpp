/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023-2024 Barcelona Supercomputing Center (BSC)
*/

#include <iostream>

#include "CUDAAccelerator.hpp"
#include "CUDADirectoryAgent.hpp"
#include "scheduling/Scheduler.hpp"

#include <api/nanos6/events.h>

CUDADirectoryAgent::CUDADirectoryAgent(CUDAAccelerator *accelerator) :
	DirectoryAgent(oss_device_cuda, accelerator->getDeviceHandler()),
	_accelerator(accelerator)
{
	_directoryStream = CUDAFunctions::createStream();
}

void CUDADirectoryAgent::memcpyFromImpl(DirectoryPage *page,
	DirectoryAgent *src,
	size_t size,
	void *srcAddress,
	void *dstAddress,
	cudaStream_t stream
	)
{
	assert(canCopyFrom(src));

	_accelerator->setActiveDevice();

	OngoingCopy copy;
	copy.destinationDevice = this->getGlobalId();
	CUDAFunctions::createEvent(copy.event);
	if (src->getType() == oss_device_host) {
		CUDAFunctions::copyMemoryAsync(dstAddress, srcAddress, size, cudaMemcpyHostToDevice, stream);
	} else {
		int srcDevice = ((CUDADirectoryAgent *)src)->_accelerator->getDeviceHandler();
		int dstDevice = _accelerator->getDeviceHandler();
		// P2P
		CUDAFunctions::copyMemoryP2PAsync(dstAddress, dstDevice, srcAddress, srcDevice, size, stream);
	}
	CUDAFunctions::recordEvent(copy.event, stream);
	page->_agentInfo[copy.destinationDevice]._copyHandler = copy.event;
	copy.page = page;

	_lock.lock();
	_pendingEvents.push_back(copy);
	_lock.unlock();
}


void CUDADirectoryAgent::memcpyFrom(DirectoryPage *page, DirectoryAgent *src, size_t size, void *srcAddress, void *dstAddress)
{
	cudaStream_t stream = _directoryStream;
	memcpyFromImpl(page, src, size, srcAddress, dstAddress, stream);
}

void CUDADirectoryAgent::memcpyFromImplicit(DirectoryPage *page, DirectoryAgent *src, size_t size, void *srcAddress, void *dstAddress, Task *task)
{
	cudaStream_t stream = task->getDeviceEnvironment().cuda.stream;
	memcpyFromImpl(page, src, size, srcAddress, dstAddress, stream);
}

void CUDADirectoryAgent::synchronizeOngoing(DirectoryPage *page, Task *task)
{
	cudaEvent_t event = (cudaEvent_t) page->_agentInfo[this->getGlobalId()]._copyHandler;
	cudaStream_t stream = task->getDeviceEnvironment().cuda.stream;

	CUDAFunctions::waitForEvent(event, stream);
}

void CUDADirectoryAgent::memcpy(DirectoryPage *page, DirectoryAgent *dst, size_t size, void *srcAddress, void *dstAddress)
{
	assert(canCopyTo(dst));

	_accelerator->setActiveDevice();
	OngoingCopy copy;
	CUDAFunctions::createEvent(copy.event);
	copy.destinationDevice = dst->getGlobalId();

	assert(dst->getType() == oss_device_host);
	CUDAFunctions::copyMemoryAsync(dstAddress, srcAddress, size, cudaMemcpyDeviceToHost, _directoryStream);
	CUDAFunctions::recordEvent(copy.event, _directoryStream);
	page->_agentInfo[copy.destinationDevice]._copyHandler = copy.event;
	copy.page = page;

	// TODO: This could be a lock-free MPSC queue.
	_lock.lock();
	_pendingEvents.push_back(copy);
	_lock.unlock();
}

void CUDADirectoryAgent::processEvents()
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

			page->lock();

			cudaEvent_t ongoingEvent = (cudaEvent_t) page->_agentInfo[copy.destinationDevice]._copyHandler;
			// If the events on the copy do not match, it means that it has been finalized by another
			// thread as part of an implicit synchronization. This can happen when a task finalization is processed
			// before ready copies, and thus a descendant task finds the copy still transitioning. In that case,
			// the descendant task will mark the copy as finalized.
			if (ongoingEvent == copy.event)
				page->notifyCopyFinalization(copy.destinationDevice);

			page->unlock();

			CUDAFunctions::destroyEvent(copy.event);
			_pendingEventsCopy.pop_front();
		}
	}
}
