/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/


#ifndef REDUCTION_INFO_HPP
#define REDUCTION_INFO_HPP

#include <functional>
#include <atomic>

#include "ReductionSpecific.hpp"
#include "DataAccessRegion.hpp"
#include "ReductionSpecific.hpp"
#include "executors/threads/CPUManager.hpp"
#include "lowlevel/PaddedSpinLock.hpp"
#include "support/Containers.hpp"


class DeviceReductionStorage;

class ReductionInfo
{
	public:

		typedef PaddedSpinLock<> spinlock_t;

		typedef boost::dynamic_bitset<> reduction_slot_set_t;

		ReductionInfo(void * address, size_t length, reduction_type_and_operator_index_t typeAndOperatorIndex,
			std::function<void(void*, void*, size_t)> initializationFunction,
			std::function<void(void*, void*, size_t)> combinationFunction);

		~ReductionInfo();

		reduction_type_and_operator_index_t getTypeAndOperatorIndex() const;

		const void * getOriginalAddress() const;

		size_t getOriginalLength() const;

		void combine();

		void releaseSlotsInUse(Task *task, ComputePlace* computePlace);

		void * getFreeSlot(Task *task, ComputePlace* computePlace);

		void incrementRegisteredAccesses();

		bool incrementUnregisteredAccesses();

		bool markAsClosed();

		bool finished();

		const DataAccessRegion& getOriginalRegion() const
		{
			return _region;
		}

	private:
		typedef Container::vector<ReductionSlot> slots_t;
		typedef Container::vector<long int> current_cpu_slot_indices_t;
		typedef Container::vector<size_t> free_slot_indices_t;

		DataAccessRegion _region;

		void * _address;

		const size_t _length;

		const size_t _paddedLength;

		reduction_type_and_operator_index_t _typeAndOperatorIndex;

		Container::map<nanos6_device_t, DeviceReductionStorage *> _deviceStorages;

		std::function<void(void*, size_t)> _initializationFunction;
		std::function<void(void*, void*, size_t)> _combinationFunction;

		std::atomic<size_t> _registeredAccesses;
		spinlock_t _lock;

		DeviceReductionStorage * allocateDeviceStorage(nanos6_device_t deviceType);
};

inline void ReductionInfo::incrementRegisteredAccesses()
{
	++_registeredAccesses;
}

inline bool ReductionInfo::incrementUnregisteredAccesses()
{
	assert(_registeredAccesses > 0);
	return (--_registeredAccesses == 0);
}

inline bool ReductionInfo::finished()
{
	return (_registeredAccesses == 0);
}

inline bool ReductionInfo::markAsClosed()
{
	return incrementUnregisteredAccesses();
}

#endif // REDUCTION_INFO_HPP
