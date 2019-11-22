/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/


#ifndef DEVICE_REDUCTION_STORAGE_HPP
#define DEVICE_REDUCTION_STORAGE_HPP

#include <vector>
#include <functional>
#include <queue>
#include <unordered_map>

#include <executors/threads/CPUManager.hpp>

#include "ReductionSpecific.hpp"
#include "DataAccessRegion.hpp"
#include "ReductionInfo.hpp"
#include "tasks/Task.hpp"

class DeviceReductionStorage
{
	public:
		DeviceReductionStorage(void * address, size_t length, size_t paddedLength,
			std::function<void(void*, size_t)> initializationFunction,
			std::function<void(void*, void*, size_t)> combinationFunction);

		virtual ~DeviceReductionStorage();

		virtual void releaseSlotsInUse(Task * task, ComputePlace * computePlace) = 0;

		virtual size_t getFreeSlotIndex(Task * task, ComputePlace * destinationComputePlace) = 0;

		virtual void * getFreeSlotStorage(Task * task, size_t slotIndex, ComputePlace * destinationComputePlace) = 0;

		virtual void combineInStorage(char * combineDestination) = 0;

	protected:
		void * _address;

		const size_t _length;

		const size_t _paddedLength;

		std::function<void(void*, size_t)> _initializationFunction;
		std::function<void(void*, void*, size_t)> _combinationFunction;
};

#endif // DEVICE_REDUCTION_STORAGE
