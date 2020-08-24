/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include "HostReductionStorage.hpp"
#include "MemoryAllocator.hpp"


HostReductionStorage::HostReductionStorage(void * address, size_t length, size_t paddedLength,
	std::function<void(void*, size_t)> initializationFunction,
	std::function<void(void*, void*, size_t)> combinationFunction) :
	DeviceReductionStorage(address, length, paddedLength, initializationFunction, combinationFunction),
	_freeSlotIndices(CPUManager::getTotalCPUs())
{
	const long nCpus = CPUManager::getTotalCPUs();
	assert(nCpus > 0);
	_slots.resize(nCpus); // Create all slots
	_currentCpuSlotIndices.resize(nCpus, -1);
};

void * HostReductionStorage::getFreeSlotStorage(__attribute__((unused)) Task * task, size_t slotIndex,
	__attribute__((unused)) ComputePlace * destinationComputePlace)
{
	assert(task != nullptr);
	assert(destinationComputePlace != nullptr);
	assert(slotIndex < _slots.size());

	slot_t& slot = _slots[slotIndex];
	assert(slot.initialized || slot.storage == nullptr);

	if (!slot.initialized) {
		// Allocate new storage
		slot.storage = MemoryAllocator::alloc(_paddedLength);

		_initializationFunction(slot.storage, _length);
		slot.initialized = true;
	}

	return slot.storage;
}

void HostReductionStorage::combineInStorage(char * combineDestination)
{
	std::lock_guard<ReductionInfo::spinlock_t> guard(_lock);
	assert(combineDestination != nullptr);

	for(size_t i = 0; i < _slots.size(); ++i) {
        slot_t& slot = _slots[i];

        if (slot.initialized) {
			assert(slot.storage != nullptr);
			assert(slot.storage != (void *) combineDestination);

			_combinationFunction((void *) combineDestination, slot.storage, _length);

			MemoryAllocator::free(slot.storage, _paddedLength);
			slot.storage = nullptr;
			slot.initialized = false;
		}
    }
}

size_t HostReductionStorage::getFreeSlotIndex(Task * task, ComputePlace * destinationComputePlace)
{
	assert(destinationComputePlace->getType() == nanos6_host_device);
	int cpuId = destinationComputePlace->getIndex();
	assert((size_t) cpuId < _currentCpuSlotIndices.size());
	long int currentSlotIndex = _currentCpuSlotIndices[cpuId];

	if (currentSlotIndex != -1) {
		// Storage already assigned to this CPU, increase counter
		// Note: Currently, this can only happen with a weakreduction task with
		// 2 or more (in_final) reduction subtasks that will be requesting storage
		// Note: Task scheduling points within reduction are currently not supported,
		// as tied tasks are not yet implemented. If supported, task counters would be
		// required to avoid the storage to be released at the end of a task while still in use

		assert(_slots[currentSlotIndex].initialized);
		return currentSlotIndex;
	}

	int freeSlotIndex = _freeSlotIndices.setFirst();
	// Can this happen??
	while (freeSlotIndex == -1)
		freeSlotIndex = _freeSlotIndices.setFirst();

	_currentCpuSlotIndices[cpuId] = freeSlotIndex;

	return freeSlotIndex;
}

void HostReductionStorage::releaseSlotsInUse(Task * task, ComputePlace * computePlace)
{
	assert(computePlace->getType() == nanos6_host_device);
	int cpuId = computePlace->getIndex();
	assert(cpuId < _currentCpuSlotIndices.size());
	long int currentSlotIndex = _currentCpuSlotIndices[cpuId];

	// Note: If access is weak and final (promoted), but had no reduction subtasks, this
	// member can be called when _currentCpuSlotIndices[task] is invalid (hasn't been used)
	if (currentSlotIndex != -1)
	{
		assert(_slots[currentSlotIndex].storage != nullptr);
		assert(_slots[currentSlotIndex].initialized);
		_freeSlotIndices.reset(currentSlotIndex);
		_currentCpuSlotIndices[cpuId] = -1;
	}
}