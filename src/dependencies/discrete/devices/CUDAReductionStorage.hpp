/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <config.h>

#if USE_CUDA
#ifndef CUDA_REDUCTION_STORAGE_HPP
#define CUDA_REDUCTION_STORAGE_HPP

#include "dependencies/discrete/DeviceReductionStorage.hpp"

class CUDAReductionStorage : public DeviceReductionStorage {
public:
	struct ReductionSlot {
		void *storage = nullptr;
		bool initialized = false;
		int deviceId = 0;
	};

	typedef std::unordered_map<Task *, size_t> assignation_map_t;
	typedef ReductionSlot slot_t;

	CUDAReductionStorage(void *address, size_t length, size_t paddedLength,
		std::function<void(void *, size_t)> initializationFunction,
		std::function<void(void *, void *, size_t)> combinationFunction);

	void *getFreeSlotStorage(Task *task, size_t slotIndex, ComputePlace *destinationComputePlace);

	void combineInStorage(void *combineDestination);

	void releaseSlotsInUse(Task *task, ComputePlace *computePlace);

	size_t getFreeSlotIndex(Task *task, ComputePlace *destinationComputePlace);

	~CUDAReductionStorage(){};

private:
	ReductionInfo::spinlock_t _lock;
	std::vector<slot_t> _slots;
	assignation_map_t _currentAssignations;
	std::vector<std::vector<size_t>> _freeSlotIndices;
};

#endif // CUDA_REDUCTION_STORAGE_HPP
#endif // USE_CUDA