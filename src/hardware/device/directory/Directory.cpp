/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#include <sys/mman.h>

#include "Directory.hpp"
#include "hardware/HardwareInfo.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "tasks/Task.hpp"

#include <DataAccess.hpp>
#include <TaskDataAccesses.hpp>

Directory *Directory::_instance;
DirectoryDevice *Directory::_hostDevice;
std::atomic<bool> Directory::_enabled;

static inline bool canRead(DirectoryPageState state)
{
	return (state == StateExclusive || state == StateModified || state == StateShared);
}

static inline bool isTransitioning(DirectoryPageState state)
{
	return (
		state == StateTransitionExclusive ||
		state == StateTransitionModified ||
		state == StateTransitionShared);
}

static inline bool canWrite(DirectoryPageState state)
{
	return (state == StateExclusive || state == StateModified);
}

bool Directory::preTaskExecution(
	DirectoryDevice *device,
	Task *task,
	nanos6_address_translation_entry_t *translationTable,
	int symbols)
{
	return _instance->processTaskAccesses(device, task, translationTable, symbols);
}

static inline bool copyPage(DirectoryDevice *src, DirectoryDevice *dst, size_t size, DirectoryPage *page, Task *task)
{
	if (dst->canCopyFrom(src)) {
		if (task != nullptr && dst->canSynchronizeImplicitely()) {
			dst->memcpyFromImplicit(page, src, size, page->_allocations[src->getId()], page->_allocations[dst->getId()], task);
			return true;
		} else {
			dst->memcpyFrom(page, src, size, page->_allocations[src->getId()], page->_allocations[dst->getId()]);
			return false;
		}
	} else {
		assert(src->canCopyTo(dst));
		src->memcpy(page, dst, size, page->_allocations[src->getId()], page->_allocations[dst->getId()]);
		return false;
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
		// Non-tracked entry
		return;
	}

	size_t pageSize = locationEntry->getPageSize();

	if (length > pageSize)
		FatalErrorHandler::fail("Cannot use access length larger than directory page size. Use multideps instead");

	int pageIndex = locationEntry->getPageIdx(location);

	DirectoryPage *page = locationEntry->getPage(pageIndex);
	page->lock();
	DirectoryPageState state = page->_states[directoryId];

	if (!canWrite(state)) {
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
				if (remoteState != StateInvalid) {
					assert(remoteState != StateTransitionShared);
					if (!found) {
						found = true;
						allocatePageIfNeeded(device, pageSize, page);
						if (!copyPage(_devices[i], device, pageSize, page, task)) {
							task->increasePredecessors(1);
							page->_pendingNotifications[directoryId].push_back(task);
						}
					}

					if (isTransitioning(remoteState))
						page->notifyCopyFinalization(i);

					page->_states[i] = StateInvalid;
				}

				assert(!isTransitioning(remoteState));
			}
		} else {
			assert(state != StateTransitionShared);
			if (device->canSynchronizeOngoingCopies()) {
				device->synchronizeOngoing(page, task);
			} else {
				task->increasePredecessors(1);
				page->_pendingNotifications[directoryId].push_back(task);
			}
		}
	}

	// Set translation
	// Technically we don't need the lock for this, but given that we already have it we'll just do it here
	translation = page->_allocations[directoryId];

	page->unlock();
}

void Directory::readAccess(DirectoryDevice *device, void *location, size_t length, Task *task, void *&translation)
{
	const int directoryId = device->getId();
	DirectoryEntry *locationEntry = getEntry((addr_t) location);
	if (!locationEntry || !locationEntry->includes(location)) {
		// Non-tracked entry
		return;
	}

	size_t pageSize = locationEntry->getPageSize();

	if (length > pageSize)
		FatalErrorHandler::fail("Cannot use access length larger than directory page size. Use multideps instead");

	int pageIndex = locationEntry->getPageIdx(location);

	DirectoryPage *page = locationEntry->getPage(pageIndex);
	page->lock();
	DirectoryPageState state = page->_states[directoryId];

	if (!canRead(state)) {
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
				if (canRead(remoteState) || (isTransitioning(remoteState) && remoteState != StateTransitionShared)) {
					found = true;

					allocatePageIfNeeded(device, pageSize, page);
					if(!copyPage(_devices[i], device, pageSize, page, task)) {
						task->increasePredecessors(1);
						page->_pendingNotifications[directoryId].push_back(task);
					}

					if (isTransitioning(remoteState)) {
						page->notifyCopyFinalization(i);
					}

					if (remoteState == StateExclusive || remoteState == StateModified) {
						page->_states[i] = StateShared;
					}

					break;
				}
			}

			if (!found)
				FatalErrorHandler::fail("Failure in D/C, not found source for read access");
		} else {
			if (device->canSynchronizeOngoingCopies()) {
				device->synchronizeOngoing(page, task);
			} else {
				task->increasePredecessors(1);
				page->_pendingNotifications[directoryId].push_back(task);
			}
		}
	}

	// Set translation
	// Technically we don't need the lock for this, but given that we already have it we'll just do it here
	translation = page->_allocations[directoryId];

	page->unlock();
}

void Directory::registerEntry(DirectoryDevice *device, void *buffer, void *virtualBuffer, size_t size, size_t pageSize)
{
	const int directoryId = device->getId();

	if (!isEnabled())
		_enabled.store(true, std::memory_order_release);

	_lock.writeLock();
	_directory.emplace(std::piecewise_construct,
		std::forward_as_tuple((addr_t) virtualBuffer),
		std::forward_as_tuple(buffer, virtualBuffer, size, pageSize, directoryId, _devices.size()));
	_lock.writeUnlock();
}

void Directory::destroyEntry(void *buffer)
{
	assert(isEnabled());
	DirectoryEntry *entry = getEntry((addr_t) buffer);
	if (!entry || !entry->includes(buffer)) {
		// Non-tracked entry
		FatalErrorHandler::fail("Double free or corruption (a.k.a tried to free a buffer not registered in the directory)");
		return;
	}

	const int homeDevice = entry->getHomeDevice();
	const int numDevices = _devices.size();
	const size_t numPages = entry->getNumPages();
	// Free every buffer except the one in the host and the home device, since those are allocated consecutively
	for (size_t i = 0; i < numPages; ++i) {
		DirectoryPage *page = entry->getPage(i);

		for (int dev = 0; dev < numDevices; ++dev) {
			if (dev == homeDevice || dev == 0)
				continue;

			if (page->_allocations[dev] != nullptr) {
				_devices[dev]->freeMemory(page->_allocations[dev], entry->getPageSize());
			}
		}
	}

	_devices[homeDevice]->freeMemory(entry->getBaseAddress(), entry->getSize());

	if (homeDevice != 0) {
		// Free host shadow region
		int res = munmap(entry->getBaseVirtualAddress(), entry->getSize());
		FatalErrorHandler::failIf(res != 0, "Failed to unmap host shadow region");
	}

	_lock.writeLock();
	_directory.erase((addr_t) entry->getBaseVirtualAddress());
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
	void *virtualBuffer = buffer;
	if (device != oss_device_host) {
		// Allocate virtual memory in the host to act as a shadow region
		// This allows using nested tasks because addresses in the host are not really translated ever
		virtualBuffer = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
		assert(virtualBuffer != MAP_FAILED);
	}

	_instance->registerEntry(deviceHandle, buffer, virtualBuffer, size, stride);

	return buffer;
}

void Directory::deviceFree(void *address)
{
	_instance->destroyEntry(address);
}

extern "C" void *oss_device_alloc(oss_device_t device, int index, size_t size, size_t access_stride)
{
	return Directory::deviceAlloc(device, index, size, access_stride);
}

extern "C"  void oss_device_free(void *address)
{
	Directory::deviceFree(address);
}

void Directory::flushEntry(DirectoryEntry *entry, Task *taskToUnlock)
{
	partiallyFlushEntry(entry, taskToUnlock, entry->getBaseVirtualAddress(), entry->getSize());
}

// MMap adreces virtuals sempre host
// Sempre es tradueixen devices! (Bueno, es pot traduïr tot sempre no passa res)
// Flags a la allocation per dir si ha d ser memòria consecutiva
// Tractar sempre x pàgines, no admetre deps de + 1 pàgina

void Directory::partiallyFlushEntry(DirectoryEntry *entry, Task *taskToUnlock, void *location, size_t length)
{
	size_t pageSize = entry->getPageSize();

	if (length % pageSize != 0)
		FatalErrorHandler::fail("Length is not multiple of declared access stride");

	const int homeDevice = entry->getHomeDevice();

	int pageIndex = entry->getPageIdx(location);

	for (int currentPage = 0; currentPage < (int) (length / pageSize); currentPage++)
	{
		DirectoryPage *page = entry->getPage(pageIndex + currentPage);
		page->lock();
		DirectoryPageState state = page->_states[homeDevice];

		// We can assume at this point every access to this region has been closed, so any transitioning
		// states that exist are because they haven't been processed by the relevant polling services yet.

		if (isTransitioning(state)) {
			page->notifyCopyFinalization(homeDevice);
			state = page->_states[homeDevice];
		}

		assert(!isTransitioning(state));
		if (state == StateModified || state == StateShared) {
			// We already have a valid copy
			state = StateExclusive;
		} else if (state == StateInvalid) {
			// We need to enqueue a copy then
			taskToUnlock->increasePredecessors(1);
			page->_states[homeDevice] = StateTransitionExclusive;
			page->_pendingNotifications[homeDevice].push_back(taskToUnlock);

			for (size_t i = 0; i < _devices.size(); ++i) {
				if (i == (size_t) homeDevice)
					continue;

				DirectoryPageState remoteState = page->_states[i];
				if (canRead(remoteState) || isTransitioning(remoteState)) {
					if (isTransitioning(remoteState)) {
						page->notifyCopyFinalization(i);
						page->_states[i] = StateInvalid;
					}
					copyPage(_devices[i], _devices[homeDevice], pageSize, page, nullptr);
					break;
				}
			}
		} else {
			assert(state == StateExclusive);
		}

		for (size_t i = 0; i < _devices.size(); ++i) {
			if (i == (size_t) homeDevice)
				continue;

			if (isTransitioning(state))
				page->notifyCopyFinalization(i);
			page->_states[i] = StateInvalid;
		}

		page->unlock();
	}
}

void Directory::performFullFlush(Task *taskToUnlock)
{
	_lock.readLock();

	for (directory_map_t::iterator it = _directory.begin(); it != _directory.end(); ++it)
		flushEntry(&(it->second), taskToUnlock);

	_lock.readUnlock();
}

void Directory::performPartialFlush(void *location, size_t length, Task *taskToUnlock)
{
	DirectoryEntry *locationEntry = getEntry((addr_t) location);
	if (!locationEntry || !locationEntry->includes(location)) {
		// Non-tracked entry
		return;
	}

	partiallyFlushEntry(locationEntry, taskToUnlock, location, length);
}

bool Directory::flush(Task *task)
{
	task->increasePredecessors();
	_instance->performFullFlush(task);
	return task->decreasePredecessors();
}

#ifdef REGIONS_DEPENDENCIES
bool Directory::processTaskAccesses(DirectoryDevice *, Task *, nanos6_address_translation_entry_t *, int)
{
	return true;
}

bool Directory::flushTaskDependencies(Task *)
{
	return true;
}

#else
bool Directory::flushTaskDependencies(Task *task)
{
	if (!isEnabled())
		return true;

	task->increasePredecessors();

	// Process task accesses, searching for accesses
	TaskDataAccesses &accessStruct = task->getDataAccesses();
	accessStruct.forAll([task](void *address, DataAccess *access) -> bool {
		// Weak accesses don't really need to be translated
		if (access->isWeak())
			return true;

		_instance->performPartialFlush(address, access->getLength(), task);

		return true;
	});

	return task->decreasePredecessors();
}

bool Directory::processTaskAccesses(DirectoryDevice *device, Task *task, nanos6_address_translation_entry_t *translationTable, int symbols)
{
	if (!isEnabled())
		return true;

	task->increasePredecessors();

	// Process task accesses, searching for accesses
	TaskDataAccesses &accessStruct = task->getDataAccesses();
	accessStruct.forAll(
		[this, device, task, symbols, translationTable]
		(void *address,	DataAccess *access) -> bool {
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
