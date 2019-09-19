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
#include <lowlevel/PaddedSpinLock.hpp>

#include <boost/dynamic_bitset.hpp>

#include "ReductionSpecific.hpp"
#include "DataAccessRegion.hpp"

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
		
		ReductionInfo(void * address, size_t length, reduction_type_and_operator_index_t typeAndOperatorIndex,
				std::function<void(void*, void*, size_t)> initializationFunction, 
				std::function<void(void*, void*, size_t)> combinationFunction);
		
		~ReductionInfo();
		
		reduction_type_and_operator_index_t getTypeAndOperatorIndex() const;
		
		const void * getOriginalAddress() const;

		size_t getOriginalLength() const;
		
		bool combine(bool canCombineToOriginalStorage);
		
		void releaseSlotsInUse(size_t virtualCpuId);
		
		size_t getFreeSlotIndex(size_t virtualCpuId);
		
		void * getFreeSlotStorage(size_t slotIndex);
		
		void makeOriginalStorageAvailable(const void * address, const size_t length);

		bool noSlotsInUse();

		void incrementRegisteredAccesses();

		bool incrementUnregisteredAccesses();

		bool markAsClosed();

		bool finished();
		
		const DataAccessRegion& getOriginalRegion() const 
		{
			return _region;
		}
		
	private:
		DataAccessRegion _region;
		
		void * _address;
		
		const size_t _length;
		
		const size_t _paddedLength;
		
		reduction_type_and_operator_index_t _typeAndOperatorIndex;
		
		std::atomic_size_t _originalStorageCombinationCounter;
		
		std::atomic_size_t _privateStorageCombinationCounter;
		
		bool _isOriginalStorageAvailable;
		std::atomic_size_t _originalStorageAvailabilityCounter;
		
		std::vector<ReductionSlot> _slots;
		std::vector<long int> _currentCpuSlotIndices;
		std::vector<size_t> _freeSlotIndices;
		// Aggregating slots are private slots used to aggregate combinations
		// when the original region is not available for combination. By now, they are not being used 
		// in discrete-simple deps.
		reduction_slot_set_t _isAggregatingSlotIndex;
		reduction_slot_set_t _privateSlotsUsed;
		
		std::function<void(void*, size_t)> _initializationFunction;
		std::function<void(void*, void*, size_t)> _combinationFunction;

		std::atomic<size_t> _registeredAccesses;
		std::atomic<size_t> _unregisteredAccesses;
		
		spinlock_t _lock;
};

inline size_t ReductionInfo::getMaxSlots()
{
	// Note: This can't become a const static member because on its definition
	// it would call 'getTotalCPUs' before the runtime is properly initialized
	// Note: '+1' when original storage is available
	return CPUManager::getTotalCPUs() + 1;
}

inline bool ReductionInfo::noSlotsInUse() 
{
	_lock.lock();
	return (_freeSlotIndices.size() == getMaxSlots());
	_lock.unlock();
}

inline void ReductionInfo::incrementRegisteredAccesses()
{
	++_registeredAccesses;
}

inline bool ReductionInfo::incrementUnregisteredAccesses()
{
	assert(_unregisteredAccesses != _registeredAccesses);
	return (++_unregisteredAccesses == _registeredAccesses);
}

inline bool ReductionInfo::finished()
{
	bool finished = (_unregisteredAccesses == _registeredAccesses);
	return finished;
}

inline bool ReductionInfo::markAsClosed()
{
	return incrementUnregisteredAccesses();
}

#endif // REDUCTION_INFO_HPP
