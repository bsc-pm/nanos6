/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "ReductionInfo.hpp"

#include <cassert>
#include <sys/mman.h>

#include <hardware/HardwareInfo.hpp>

#include <MemoryAllocator.hpp>

#include <InstrumentReductions.hpp>

#include <executors/threads/WorkerThread.hpp>
#include <InstrumentThreadInstrumentationContextImplementation.hpp>

ReductionInfo::ReductionInfo(DataAccessRegion region, reduction_type_and_operator_index_t typeAndOperatorIndex,
		std::function<void(void*, void*, size_t)> initializationFunction, std::function<void(void*, void*, size_t)> combinationFunction) :
	_region(region),
	_paddedRegionSize(((_region.getSize() + HardwareInfo::getCacheLineSize() - 1)/HardwareInfo::getCacheLineSize())*HardwareInfo::getCacheLineSize()),
	_typeAndOperatorIndex(typeAndOperatorIndex),
	_originalStorageCombinationCounter(_region.getSize()),
	_initializationFunction(std::bind(initializationFunction, std::placeholders::_1, _region.getStartAddress(), std::placeholders::_2)),
	_combinationFunction(combinationFunction)
{
	const long nCpus = CPUManager::getTotalCPUs();
	assert(nCpus > 0);
	
	const size_t maxSlots = getMaxSlots();
	_slots.reserve(maxSlots);
	_freeSlotIndices.reserve(maxSlots);
	_currentCpuSlotIndices.resize(nCpus, -1);
}

ReductionInfo::~ReductionInfo()
{
#ifndef NDEBUG
	for (int slotIndex : _currentCpuSlotIndices)
		assert(slotIndex == -1);
#endif
	
	void *originalRegionStorage = _region.getStartAddress();
	for (ReductionSlot& slot : _slots)
	{
		if (slot.storage != originalRegionStorage) {
			assert(slot.storage != nullptr);
			MemoryAllocator::free(slot.storage, _paddedRegionSize);
		}
	}
}

reduction_type_and_operator_index_t ReductionInfo::getTypeAndOperatorIndex() const {
	return _typeAndOperatorIndex;
}

const DataAccessRegion& ReductionInfo::getOriginalRegion() const {
	return _region;
}

size_t ReductionInfo::getFreeSlotIndex(size_t virtualCpuId) {
	__attribute__((unused)) const long nCpus = CPUManager::getTotalCPUs();
	assert(nCpus > 0);
	assert(virtualCpuId < (size_t)nCpus);
	assert(virtualCpuId < _currentCpuSlotIndices.size());
	
	long int currentCpuSlotIndex = _currentCpuSlotIndices[virtualCpuId];
	
	if (currentCpuSlotIndex != -1) {
		// Storage already assigned to this CPU, increase counter
		// Note: Currently, this can only happen with a weakreduction task with
		// 2 or more (in_final) reduction subtasks that will be requesting storage
		// Note: Task scheduling points within reduction are currently not supported,
		// as tied tasks are not yet implemented. If supported, task counters would be
		// required to avoid the storage to be released at the end of a task while still in use

		assert(_slots[currentCpuSlotIndex].initialized);
		return currentCpuSlotIndex;
	}
	
	// Lock required to access _freeSlotIndices simultaneously
	_lock.lock();
	size_t freeSlotIndex;
	if (_freeSlotIndices.size() > 0) {
		// Reuse free slot in pool
		freeSlotIndex = _freeSlotIndices.back();
		_freeSlotIndices.pop_back();
		
		_lock.unlock();
	}
	else {
		// Allocate new storage
		Instrument::enterAllocatePrivateReductionStorage(
			/* reductionInfo */ *this
		);
		
		FatalErrorHandler::failIf(_slots.size() > getMaxSlots(), "Maximum number of private storage slots reached");
		freeSlotIndex = _slots.size();
		_slots.emplace_back();
		ReductionSlot& newSlot = _slots.back();
		
		_lock.unlock();
		
		newSlot.storage = MemoryAllocator::alloc(_paddedRegionSize);
		
		Instrument::exitAllocatePrivateReductionStorage(
			/* reductionInfo */ *this,
			DataAccessRegion(newSlot.storage, _region.getSize())
		);
	}
	_currentCpuSlotIndices[virtualCpuId] = freeSlotIndex;
	
	return freeSlotIndex;
}

DataAccessRegion ReductionInfo::getFreeSlotStorage(size_t slotIndex) {
	assert(slotIndex < _slots.size());
	
	ReductionSlot& slot = _slots[slotIndex];
	assert(slot.storage != nullptr);
	
	if (!slot.initialized) {
		Instrument::enterInitializePrivateReductionStorage(
			/* reductionInfo */ *this,
			DataAccessRegion(slot.storage, _region.getSize())
		);
		
		_initializationFunction(slot.storage, _region.getSize());
		slot.initialized = true;
		
		Instrument::exitInitializePrivateReductionStorage(
			/* reductionInfo */ *this,
			DataAccessRegion(slot.storage, _region.getSize())
		);
	}
	
	return DataAccessRegion(slot.storage, _region.getSize());
}

bool ReductionInfo::combineRegion(const DataAccessRegion& region, const reduction_slot_set_t& accessedSlots) {
	assert(accessedSlots.size() > 0);
	
	void *originalRegionStorage = region.getStartAddress();
	
	for (size_t i = 0; i < accessedSlots.size(); i++) {
		ReductionSlot& slot = _slots[i];
		if (accessedSlots[i] && (slot.storage != _region.getStartAddress()))
		{
			void *privateStorage = ((char*)slot.storage) + ((char*)region.getStartAddress() - (char*)_region.getStartAddress());
			
			Instrument::enterCombinePrivateReductionStorage(
				/* reductionInfo */ *this,
				DataAccessRegion(privateStorage, region.getSize()),
				DataAccessRegion(originalRegionStorage, region.getSize())
			);
			
			_combinationFunction(originalRegionStorage, privateStorage, region.getSize());
			
			Instrument::exitCombinePrivateReductionStorage(
				/* reductionInfo */ *this,
				DataAccessRegion(privateStorage, region.getSize()),
				DataAccessRegion(originalRegionStorage, region.getSize())
			);
		}
	}
	
	_originalStorageCombinationCounter -= region.getSize();
	
	return _originalStorageCombinationCounter == 0;
}

void ReductionInfo::releaseSlotsInUse(size_t virtualCpuId) {
	std::lock_guard<spinlock_t> guard(_lock);
	
	long int currentCpuSlotIndex = _currentCpuSlotIndices[virtualCpuId];
	// Note: If access is weak and final (promoted), but had no reduction subtasks, this
	// member can be called when _currentCpuSlotIndices[virtualCpuId] is invalid (hasn't been used)
	if (currentCpuSlotIndex != -1)
	{
		assert(_slots[currentCpuSlotIndex].storage != nullptr);
		assert(_slots[currentCpuSlotIndex].initialized);
		_freeSlotIndices.push_back(currentCpuSlotIndex);
		_currentCpuSlotIndices[virtualCpuId] = -1;
	}
}
