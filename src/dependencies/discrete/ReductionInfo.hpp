/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/


#ifndef REDUCTION_INFO_HPP
#define REDUCTION_INFO_HPP

#include <functional>
#include <atomic>
#include "DataAccessRegion.hpp"
#include "ReductionSpecific.hpp"
#include "executors/threads/CPUManager.hpp"
#include "lowlevel/PaddedSpinLock.hpp"
#include "support/Containers.hpp"


class DeviceReductionStorage;

class ReductionInfo {
private:
	typedef PaddedSpinLock<> spinlock_t;
	typedef boost::dynamic_bitset<> reduction_slot_set_t;

	DataAccessRegion _region;

	void *_address;

	const size_t _length;

	const size_t _paddedLength;

	reduction_type_and_operator_index_t _typeAndOperatorIndex;

	DeviceReductionStorage *_deviceStorages[nanos6_device_type_num];

	std::function<void(void *, void *, size_t)> _initializationFunction;
	std::function<void(void *, void *, size_t)> _combinationFunction;

	std::atomic<size_t> _registeredAccesses;
	spinlock_t _lock;

	DeviceReductionStorage *allocateDeviceStorage(nanos6_device_t deviceType);

public:

	ReductionInfo(void *address, size_t length, reduction_type_and_operator_index_t typeAndOperatorIndex,
		std::function<void(void *, void *, size_t)> initializationFunction,
		std::function<void(void *, void *, size_t)> combinationFunction);

	~ReductionInfo();

	inline reduction_type_and_operator_index_t getTypeAndOperatorIndex() const
	{
		return _typeAndOperatorIndex;
	}

	inline const void *getOriginalAddress() const
	{
		return _address;
	}

	size_t getOriginalLength() const
	{
		return _length;
	}

	void combine();

	void releaseSlotsInUse(Task *task, ComputePlace *computePlace);

	void *getFreeSlot(Task *task, ComputePlace *computePlace);

	inline void incrementRegisteredAccesses()
	{
		++_registeredAccesses;
	}

	inline bool incrementUnregisteredAccesses()
	{
		assert(_registeredAccesses > 0);
		return (--_registeredAccesses == 0);
	}

	inline bool markAsClosed()
	{
		return incrementUnregisteredAccesses();
	}

	bool finished()
	{
		return (_registeredAccesses == 0);
	}

	const DataAccessRegion &getOriginalRegion() const
	{
		return _region;
	}
};

#endif // REDUCTION_INFO_HPP
