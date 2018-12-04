/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef REDUCTION_INFO_HPP
#define REDUCTION_INFO_HPP

#include <vector>
#include <functional>
#include <atomic>

#include <executors/threads/CPUManager.hpp>
#include <DataAccessRegion.hpp>
#include <lowlevel/PaddedSpinLock.hpp>

#include <boost/dynamic_bitset.hpp>

#include "ReductionSpecific.hpp"

class ReductionInfo
{
	public:
		
		typedef PaddedSpinLock<> spinlock_t;
		
		struct ReductionSlot {
			void *storage = nullptr;
			bool initialized = false;
		};
		
		typedef boost::dynamic_bitset<> reduction_slot_set_t;
		
		inline static size_t getMaxSlots();
		
		ReductionInfo(DataAccessRegion region, reduction_type_and_operator_index_t typeAndOperatorIndex,
				std::function<void(void*, void*, size_t)> initializationFunction, std::function<void(void*, void*, size_t)> combinationFunction);
		
		~ReductionInfo();
		
		reduction_type_and_operator_index_t getTypeAndOperatorIndex() const;
		
		const DataAccessRegion& getOriginalRegion() const;
		
		bool combineRegion(const DataAccessRegion& subregion, reduction_slot_set_t& accessedSlots, bool canCombineToOriginalStorage);
		
		void releaseSlotsInUse(size_t virtualCpuId);
		
		size_t getFreeSlotIndex(size_t virtualCpuId);
		
		DataAccessRegion getFreeSlotStorage(size_t slotIndex);
		
		void makeOriginalStorageRegionAvailable(const DataAccessRegion &region);
		
	private:
		
		DataAccessRegion _region;
		
		const size_t _paddedRegionSize;
		
		reduction_type_and_operator_index_t _typeAndOperatorIndex;
		
		std::atomic_size_t _originalStorageCombinationCounter;
		
		std::atomic_size_t _privateStorageCombinationCounter;
		
		bool _isOriginalStorageAvailable;
		std::atomic_size_t _originalStorageAvailabilityCounter;
		
		std::vector<ReductionSlot> _slots;
		std::vector<long int> _currentCpuSlotIndices;
		std::vector<size_t> _freeSlotIndices;
		// Aggregating slots are private slots used to aggregate combinations
		// when the original region is not available for combination
		reduction_slot_set_t _isAggregatingSlotIndex;
		
		std::function<void(void*, size_t)> _initializationFunction;
		std::function<void(void*, void*, size_t)> _combinationFunction;
		
		spinlock_t _lock;
};

inline size_t ReductionInfo::getMaxSlots()
{
	// Note: This can't become a const static member because on its definition
	// it would call 'getTotalCPUs' before the runtime is properly initialized
	// Note: '+1' when original storage is available
	return CPUManager::getTotalCPUs() + 1;
}

#endif // REDUCTION_INFO_HPP
