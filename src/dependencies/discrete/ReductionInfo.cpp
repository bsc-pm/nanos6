/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include "ReductionInfo.hpp"

#include <cassert>
#include <sys/mman.h>

#include <nanos6/reductions.h>

#include <hardware/HardwareInfo.hpp>

#include <MemoryAllocator.hpp>


#include <executors/threads/WorkerThread.hpp>

ReductionInfo::ReductionInfo(void * address, size_t length, reduction_type_and_operator_index_t typeAndOperatorIndex,
		std::function<void(void*, void*, size_t)> initializationFunction, std::function<void(void*, void*, size_t)> combinationFunction) :
	_address(address),
	_length(length),
	_paddedLength(((length + HardwareInfo::getCacheLineSize() - 1)/HardwareInfo::getCacheLineSize())*HardwareInfo::getCacheLineSize()),
	_typeAndOperatorIndex(typeAndOperatorIndex),
	_originalStorageCombinationCounter(length),
	_privateStorageCombinationCounter(length),
	_isOriginalStorageAvailable(false), _originalStorageAvailabilityCounter(length),
	_initializationFunction(std::bind(initializationFunction, std::placeholders::_1, address, std::placeholders::_2)),
	_combinationFunction(combinationFunction),
	_registeredAccesses(2)
{
	const long nCpus = CPUManager::getTotalCPUs();
	assert(nCpus > 0);
	assert(_paddedLength >= _length);
	const size_t maxSlots = getMaxSlots();
	_slots.reserve(maxSlots);
	_freeSlotIndices.reserve(maxSlots);
	_currentCpuSlotIndices.resize(nCpus, -1);
	// Not being used in discrete deps.
	_isAggregatingSlotIndex.resize(maxSlots);
	_privateSlotsUsed.resize(maxSlots);
	_privateSlotsUsed.reset();
}

ReductionInfo::~ReductionInfo()
{
	assert(_registeredAccesses == 0);
	assert(_originalStorageCombinationCounter == 0);
#ifndef NDEBUG
	for (int slotIndex : _currentCpuSlotIndices)
		assert(slotIndex == -1);
#endif

	void *originalStorage = _address;
	for (size_t i = 0; i < _slots.size(); ++i) {
		ReductionSlot& slot = _slots[i];
		if (slot.storage != originalStorage) {
			assert(!_isAggregatingSlotIndex[i] || slot.storage != nullptr);

			if (slot.storage != nullptr) {
				assert(slot.initialized);
				MemoryAllocator::free(slot.storage, _paddedLength);
#ifndef NDEBUG
				slot.storage = nullptr;
				slot.initialized = false;
#endif
			}
		}
	}
}

reduction_type_and_operator_index_t ReductionInfo::getTypeAndOperatorIndex() const
{
	return _typeAndOperatorIndex;
}

const void * ReductionInfo::getOriginalAddress() const
{
	return _address;
}

size_t ReductionInfo::getOriginalLength() const
{
	return _length;
}

size_t ReductionInfo::getFreeSlotIndex(size_t virtualCpuId)
{
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
	}
	else {
		FatalErrorHandler::failIf(_slots.size() > getMaxSlots() + (_isOriginalStorageAvailable ? 0 : -1),
				"Maximum number of private storage slots reached");
		freeSlotIndex = _slots.size();
		_slots.emplace_back();
		_privateSlotsUsed.set(freeSlotIndex);
	}
	_lock.unlock();

	_currentCpuSlotIndices[virtualCpuId] = freeSlotIndex;

	return freeSlotIndex;
}

void * ReductionInfo::getFreeSlotStorage(size_t slotIndex)
{
#ifndef NDEBUG
	_lock.lock();
	assert(slotIndex < _slots.size());
	_lock.unlock();
#endif

	ReductionSlot& slot = _slots[slotIndex];
	assert(slot.initialized || slot.storage == nullptr);

	if (!slot.initialized) {
		// Allocate new storage
		slot.storage = MemoryAllocator::alloc(_paddedLength);

		_initializationFunction(slot.storage, _length);
		slot.initialized = true;
	}

	return slot.storage;
}

namespace {
	bool isBuiltinReduction(reduction_type_and_operator_index_t typeAndOperatorIndex)
	{
		assert((typeAndOperatorIndex != 0) && "Unknown reduction type and operator");
		return (typeAndOperatorIndex >= RED_TYPE_CHAR)
			&& (typeAndOperatorIndex < NUM_RED_TYPES)
			&& (typeAndOperatorIndex%1000 < NUM_RED_OPS);
	}
};

void ReductionInfo::makeOriginalStorageAvailable(__attribute__((unused)) const void * address, const size_t length)
{
	_originalStorageAvailabilityCounter -= length;

	if ((_originalStorageAvailabilityCounter == 0)
			&& isBuiltinReduction(_typeAndOperatorIndex)) {
		std::lock_guard<spinlock_t> guard(_lock);
		// Add original region to reduction slot pool
		// Note: Disabled for UDRs, as might be initialized with 'oss_orig'
		size_t freeSlotIndex = _slots.size();
		_slots.emplace_back();
		ReductionSlot& slot = _slots.back();
		assert(_address != nullptr);
		slot.storage = _address;
		slot.initialized = true;
		_freeSlotIndices.push_back(freeSlotIndex);
		_isOriginalStorageAvailable = true;
	}
}

bool ReductionInfo::combine(bool canCombineToOriginalStorage)
{
	reduction_slot_set_t &accessedSlots = _privateSlotsUsed;
	assert(accessedSlots.size() > 0);

	char *originalAddress = (char*)_address;
	char *originalSubregionAddress = (char*)_address;
	ptrdiff_t originalSubregionOffset = originalSubregionAddress - originalAddress;

	// Not being used in discrete deps. That's why there is an assert(0).
	// Select aggregating private slot
	reduction_slot_set_t::size_type aggregatingSlotIndex = reduction_slot_set_t::npos;
	if (!canCombineToOriginalStorage) {
		assert(0);

		std::lock_guard<spinlock_t> guard(_lock);
		// Try to pick one accessed slot that is already an aggregating slot
		reduction_slot_set_t candidateAggregatingSlots = accessedSlots & _isAggregatingSlotIndex;

		aggregatingSlotIndex = candidateAggregatingSlots.find_first();

		if (aggregatingSlotIndex == reduction_slot_set_t::npos) {
			// Add new aggregating slot
			aggregatingSlotIndex = accessedSlots.find_first();
			assert(aggregatingSlotIndex != reduction_slot_set_t::npos);
			assert(_slots[aggregatingSlotIndex].storage != originalAddress);

			_isAggregatingSlotIndex.set(aggregatingSlotIndex);
		}
	}

	assert(canCombineToOriginalStorage || (aggregatingSlotIndex != reduction_slot_set_t::npos));
	char *targetRegionAddress = canCombineToOriginalStorage?
		originalAddress : (char*)_slots[aggregatingSlotIndex].storage;
	assert(targetRegionAddress != nullptr);
	char *targetStorage = targetRegionAddress + originalSubregionOffset;

	reduction_slot_set_t::size_type accessedSlotIndex = accessedSlots.find_first();
	while (accessedSlotIndex < reduction_slot_set_t::npos) {
		ReductionSlot& slot = _slots[accessedSlotIndex];
		assert(accessedSlots[accessedSlotIndex]);
		if (slot.storage != targetRegionAddress) {
			char *privateStorage = ((char*)slot.storage) + originalSubregionOffset;

			if(privateStorage != nullptr)
				_combinationFunction(targetStorage, privateStorage, _length);
		}

		accessedSlotIndex = accessedSlots.find_next(accessedSlotIndex);
	}

	// Update 'accessedSlots', preparing the combination of the
	// 'aggregatingSlot' to the original region for this subregion
	if (!canCombineToOriginalStorage) {
		assert(_privateStorageCombinationCounter > 0);
		accessedSlots.reset();
		accessedSlots.set(aggregatingSlotIndex);
	}

	_privateStorageCombinationCounter -= _length;
	if (_privateStorageCombinationCounter == 0) {
#ifndef NDEBUG
		for (int slotIndex : _currentCpuSlotIndices)
			assert(slotIndex == -1);

		// At this point slots shouldn't be requested anymore
		_freeSlotIndices.clear();
#endif

		// Note: This code is only executed when all private slots have been *completely*
		// combined into aggregation slots.
		// And thus, it can't be concurrently executed with running reduction tasks,
		// only with other combinations to the original region (for a distinct access)

		_lock.lock();
		// Note: '_slots' size can still change, as original region can still
		// be made available as a slot
		size_t numSlots = _slots.size();
		_lock.unlock();

		// Note: '_slots' elements can't be erased, as positional indices kept at
		// other structures would be messed up
		for (size_t i = 0; i < numSlots; i++) {
			ReductionSlot &slot = _slots[i];
			// Not being used in discrete deps. That's why there is an assert(0).
			if (_isAggregatingSlotIndex[i]) {
				assert(0);
				// Keep slots containing aggregated contributions
				assert(slot.storage != originalAddress);
			}
			else if (slot.storage != originalAddress) {
				// Non-aggregating private slots can be deallocated and disabled
				assert(slot.storage != nullptr);
				MemoryAllocator::free(slot.storage, _paddedLength);

				// Clear slot content so that we can later detect deallocation has been done
				slot.storage = nullptr;
				slot.initialized = false;
			}
			else {
				// Original storage shouldn't be used anymore either
#ifndef NDEBUG
				slot.storage = nullptr;
				slot.initialized = false;
#endif
			}
		}
	}

	if (canCombineToOriginalStorage) {
		_originalStorageCombinationCounter -= _length;
	}

	return _originalStorageCombinationCounter == 0;
}

void ReductionInfo::releaseSlotsInUse(size_t virtualCpuId)
{
	std::lock_guard<spinlock_t> guard(_lock);

	long int currentCpuSlotIndex = _currentCpuSlotIndices[virtualCpuId];
	// Note: If access is weak and final (promoted), but had no reduction subtasks, this
	// member can be called when _currentCpuSlotIndices[virtualCpuId] is invalid (hasn't been used)
	//assert(currentCpuSlotIndex != -1);
	if (currentCpuSlotIndex != -1)
	{
		assert(_slots[currentCpuSlotIndex].storage != nullptr);
		assert(_slots[currentCpuSlotIndex].initialized);
		_freeSlotIndices.push_back(currentCpuSlotIndex);
		_currentCpuSlotIndices[virtualCpuId] = -1;
	}
}

size_t ReductionInfo::getPaddedLength() const {
	return _paddedLength;
}

