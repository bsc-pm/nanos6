/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#include "Directory.hpp"
#include "hardware/HardwareInfo.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "tasks/Task.hpp"

#include <DataAccess.hpp>
#include <TaskDataAccesses.hpp>

Directory *Directory::_instance;
DirectoryDevice *Directory::_hostDevice;

static inline bool canRead(DirectoryPageState state)
{
	return (state == StateExclusive || state == StateModified || state == StateShared);
}

static inline bool isTransitioning(DirectoryPageState state)
{
	return (state == StateTransitionExclusive || state == StateTransitionModified || state == StateTransitionShared);
}

static inline bool canWrite(DirectoryPageState state)
{
	return (state == StateExclusive || state == StateModified);
}

bool Directory::preTaskExecution(DirectoryDevice *device, Task *task, nanos6_address_translation_entry_t *translationTable, int symbols)
{
	return _instance->processTaskAccesses(device, task, translationTable, symbols);
}

static inline void copyPage(DirectoryDevice *src, DirectoryDevice *dst, size_t size, DirectoryPage *page)
{
	if (src->canCopyTo(dst)) {
		src->memcpy(page, dst, size, page->_allocations[src->getId()], page->_allocations[dst->getId()]);
	} else {
		assert(dst->canCopyFrom(src));
		dst->memcpyFrom(page, src, size, page->_allocations[src->getId()], page->_allocations[dst->getId()]);
	}
}

static inline void allocatePageIfNeeded(DirectoryDevice *device, size_t size, DirectoryPage *page)
{
	const int directoryId = device->getId();
	if (page->_allocations[directoryId] == nullptr) {
		page->_allocations[directoryId] = device->allocateMemory(size);
		assert(page->_allocations[directoryId]);
	}
}

void Directory::readWriteAccess(DirectoryDevice *device, void *location, size_t length, Task *task, void *&translation)
{
	const int directoryId = device->getId();
	DirectoryEntry *locationEntry = getEntry((addr_t) location);
	if (!locationEntry || !locationEntry->includes(location)) {
		if (!device->isHost())
			FatalErrorHandler::fail("Implicit host allocations are not supported");

		// Non-tracked entry
		return;
	}

	size_t pageSize = locationEntry->getPageSize();

	if (length % pageSize != 0)
		FatalErrorHandler::fail("Implicit host allocations are not supported");

	int pageIndex = locationEntry->getPageIdx(location);

	for (int currentPage = 0; currentPage < (int) (length / pageSize); currentPage++)
	{
		DirectoryPage *page = locationEntry->getPage(pageIndex + currentPage);
		page->lock();
		DirectoryPageState state = page->_states[directoryId];

		if (!canWrite(state)) {
			// Add the task to be notified later
			task->increasePredecessors(1);
			if (!isTransitioning(state)) {
				// We have to do the following:
				// - Search where the actual data is located
				// - Enqueue the needed operations to make the transition
				// - Add ourselves to the notifications

				bool found = false;

				if (canRead(state)) {
					// We already have the page
					page->_states[directoryId] = StateModified;
					found = true;
				} else {
					page->_states[directoryId] = StateTransitionModified;
				}

				for (size_t i = 0; i < _devices.size(); ++i) {
					if (i == (size_t) directoryId)
						continue;

					DirectoryPageState remoteState = page->_states[i];
					if (canRead(remoteState)) {
						if (!found) {
							found = true;
							allocatePageIfNeeded(device, pageSize, page);
							copyPage(_devices[i], device, pageSize, page);
							page->_pendingNotifications[directoryId].push_back(task);
						}

						page->_states[i] = StateInvalid;
					}

					assert(!isTransitioning(remoteState));
				}

				if (!found)
					FatalErrorHandler::fail("Failure in D/C, not found source for read access");
			} else {
				assert(state != StateTransitionShared);
				page->_pendingNotifications[directoryId].push_back(task);
			}
		}

		// Set translation if on first page
		// Technically we don't need the lock for this, but given that we already have it we'll just do it here
		if (currentPage == 0)
			translation = page->_allocations[directoryId];

		page->unlock();
	}
}

void Directory::readAccess(DirectoryDevice *device, void *location, size_t length, Task *task, void *&translation)
{
	const int directoryId = device->getId();
	DirectoryEntry *locationEntry = getEntry((addr_t) location);
	if (!locationEntry || !locationEntry->includes(location)) {
		if (!device->isHost())
			FatalErrorHandler::fail("Implicit host allocations are not supported");

		// Non-tracked entry
		return;
	}

	size_t pageSize = locationEntry->getPageSize();

	if (length % pageSize != 0)
		FatalErrorHandler::fail("Implicit host allocations are not supported");

	int pageIndex = locationEntry->getPageIdx(location);

	for (int currentPage = 0; currentPage < (int) (length / pageSize); currentPage++)
	{
		DirectoryPage *page = locationEntry->getPage(pageIndex + currentPage);
		page->lock();
		DirectoryPageState state = page->_states[directoryId];

		if (!canRead(state)) {
			// Add the task to be notified later
			task->increasePredecessors(1);
			page->_pendingNotifications[directoryId].push_back(task);

			if (!isTransitioning(state)) {
				// We have to do the following:
				// - Search where the actual data is located
				// - Enqueue the needed operations to make the transition
				// - Add ourselves to the notifications

				bool found = false;
				page->_states[directoryId] = StateTransitionShared;

				for (size_t i = 0; i < _devices.size(); ++i) {
					if (i == (size_t) directoryId)
						continue;

					DirectoryPageState remoteState = page->_states[i];
					if (canRead(remoteState)) {
						found = true;

						allocatePageIfNeeded(device, pageSize, page);
						copyPage(_devices[i], device, pageSize, page);

						if (remoteState == StateExclusive || remoteState == StateModified) {
							page->_states[i] = StateShared;
						}

						break;
					}
				}

				if (!found)
					FatalErrorHandler::fail("Failure in D/C, not found source for read access");
			}
		}

		// Set translation if on first page
		// Technically we don't need the lock for this, but given that we already have it we'll just do it here
		if (currentPage == 0)
			translation = page->_allocations[directoryId];

		page->unlock();
	}
}

void Directory::registerEntry(DirectoryDevice *device, void *buffer, size_t size, size_t pageSize)
{
	const int directoryId = device->getId();

	_lock.writeLock();
	std::pair<Container::map<addr_t, DirectoryEntry>::iterator, bool> emplaced =
		_directory.emplace(std::piecewise_construct,
			std::forward_as_tuple((addr_t) buffer),
			std::forward_as_tuple(buffer, size, pageSize, directoryId, _devices.size()));
	_lock.writeUnlock();
}

void *Directory::deviceAlloc(oss_device_t device, int index, size_t size, size_t stride)
{
	DirectoryDevice *deviceHandle = nullptr;

	if (device == oss_device_host) {
		deviceHandle = _hostDevice;
	} else if (device == oss_device_cuda) {
		DeviceInfo *cudaDeviceInfo = HardwareInfo::getDeviceInfo(nanos6_cuda_device);
		assert(cudaDeviceInfo);
		deviceHandle = cudaDeviceInfo->getDirectoryDevice(index);
		assert(deviceHandle);
	} else {
		FatalErrorHandler::fail("Unsupported device");
	}

	void *buffer = deviceHandle->allocateMemory(size);
	_instance->registerEntry(deviceHandle, buffer, size, stride);

	return buffer;
}

extern "C" void *oss_device_alloc(oss_device_t device, int index, size_t size, size_t access_stride)
{
	return Directory::deviceAlloc(device, index, size, access_stride);
}

extern "C"  void oss_device_free(void *)
{

}

#ifdef REGIONS_DEPENDENCIES
bool Directory::processTaskAccesses(DirectoryDevice *, Task *, nanos6_address_translation_entry_t *, int)
{
	return true;
}
#else
bool Directory::processTaskAccesses(DirectoryDevice *device, Task *task, nanos6_address_translation_entry_t *translationTable, int symbols)
{
	task->increasePredecessors();

	// Process task accesses, searching for accesses
	TaskDataAccesses &accessStruct = task->getDataAccesses();
	accessStruct.forAll([this, device, task, symbols, translationTable](void *address, DataAccess *access) -> bool {
		void *translation = nullptr;

		// Weak accesses don't really need to be translated
		if (access->isWeak())
			return true;

		// TODO other types
		if (access->getType() == READ_ACCESS_TYPE)
			this->readAccess(device, address, access->getLength(), task, translation);
		else
			this->readWriteAccess(device, address, access->getLength(), task, translation);

		if (translation != nullptr) {
			for (int j = 0; j < symbols; ++j) {
				if (access->isInSymbol(j)) {
					translationTable[j] = {(size_t)address, (size_t)translation};
				}
			}
		}

		return true;
	});

	return task->decreasePredecessors();
}
#endif // REGIONS_DEPENDENCIES
