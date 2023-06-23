/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2023 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <iostream>

#include "CUDAReductionStorage.hpp"
#include "MemoryAllocator.hpp"
#include "hardware/device/cuda/CUDAAccelerator.hpp"
#include "hardware/device/cuda/CUDADeviceInfo.hpp"
#include "hardware/device/cuda/CUDAFunctions.hpp"
#include "hardware/hwinfo/DeviceInfo.hpp"

CUDAReductionStorage::CUDAReductionStorage(void *address, size_t length, size_t paddedLength,
	std::function<void(void *, void *, size_t)> initializationFunction,
	std::function<void(void *, void *, size_t)> combinationFunction) :
	DeviceReductionStorage(address, length, paddedLength, initializationFunction, combinationFunction)
{
	CUDADeviceInfo *deviceInfoGPU = (CUDADeviceInfo *)HardwareInfo::getDeviceInfo(nanos6_cuda_device);
	assert(deviceInfoGPU != nullptr);
	int numGPUs = deviceInfoGPU->getComputePlaceCount();
	assert(numGPUs != 0);
	_freeSlotIndices.resize(numGPUs, std::vector<size_t>());
}

void *CUDAReductionStorage::getFreeSlotStorage(
	__attribute__((unused)) Task *task,
	size_t slotIndex,
	ComputePlace *destinationComputePlace)
{
	int deviceId = destinationComputePlace->getIndex();
	int oldDeviceId = 0;
	slot_t slot;

	{
		std::lock_guard<ReductionInfo::spinlock_t> guard(_lock);
		assert(slotIndex < _slots.size());
		slot = _slots[slotIndex];
	}

	assert(slot.initialized || slot.storage == nullptr);
	assert(!slot.initialized || slot.deviceId == deviceId);

	if (!slot.initialized) {
		// We assume we're coming from a polling task with a valid CUDA context
		oldDeviceId = CUDAFunctions::getActiveDevice();

		// Set the GPU we're using if needed
		if (oldDeviceId != deviceId) {
			CUDAFunctions::setActiveDevice(deviceId);
		}

		// Allocate new CUDA Storage
		slot.storage = CUDAFunctions::malloc(_paddedLength);

		// Allocate temporal host storage. This is needed to transfer the initialized
		// memory region with the neutral value towards the GPU
		void *tmpStorage = CUDAFunctions::mallocHost(_paddedLength);
		assert(tmpStorage != nullptr);

		_initializationFunction(tmpStorage, _address, _length);

		// Transfer initialized value to GPU
		CUDAFunctions::copyMemory(slot.storage, tmpStorage, _paddedLength, cudaMemcpyHostToDevice);
		CUDAFunctions::freeHost(tmpStorage);

		slot.initialized = true;
		slot.deviceId = deviceId;

		// Restore old GPU value if needed
		if (oldDeviceId != deviceId) {
			CUDAFunctions::setActiveDevice(oldDeviceId);
		}

		{
			std::lock_guard<ReductionInfo::spinlock_t> guard(_lock);
			_slots[slotIndex] = slot;
		}
	}

	return slot.storage;
}

void CUDAReductionStorage::combineInStorage(void *combineDestination)
{
	std::lock_guard<ReductionInfo::spinlock_t> guard(_lock);

	assert(combineDestination != nullptr);

	if (_slots.size() == 0)
		return;

	// Temporal storage on which to transfer device memory
	void *tmpStorage = CUDAFunctions::mallocHost(_paddedLength);
	assert(tmpStorage != nullptr);

	// Assume we are inside a valid CUDA context
	int oldDeviceId = CUDAFunctions::getActiveDevice();
	int lastDevice = oldDeviceId;

	for (size_t i = 0; i < _slots.size(); ++i) {
		slot_t &slot = _slots[i];

		assert(slot.initialized);
		assert(slot.storage != nullptr);
		assert(slot.storage != combineDestination);

		if (slot.deviceId != lastDevice) {
			CUDAFunctions::setActiveDevice(slot.deviceId);
			lastDevice = slot.deviceId;
		}

		// Transfer from GPU to host
		CUDAFunctions::copyMemory(tmpStorage, slot.storage, _paddedLength, cudaMemcpyDeviceToHost);

		// Combine from temporal host storage
		_combinationFunction(combineDestination, tmpStorage, _length);

		// Free slot
		CUDAFunctions::free(slot.storage);

		slot.storage = nullptr;
		slot.initialized = false;
	}

	if (oldDeviceId != lastDevice) {
		CUDAFunctions::setActiveDevice(oldDeviceId);
	}

	CUDAFunctions::freeHost(tmpStorage);
}

size_t CUDAReductionStorage::getFreeSlotIndex(Task *task, ComputePlace *destinationComputePlace)
{
	std::lock_guard<ReductionInfo::spinlock_t> guard(_lock);

	assert(destinationComputePlace->getType() == nanos6_cuda_device);
	assignation_map_t::iterator itSlot = _currentAssignations.find(task);

	int gpuId = destinationComputePlace->getIndex();
	assert((size_t)gpuId < _freeSlotIndices.size());

	if (itSlot != _currentAssignations.end()) {
		long int currentSlotIndex = itSlot->second;

		assert(_slots[currentSlotIndex].initialized);
		return currentSlotIndex;
	}

	// Lock required to access _freeSlotIndices simultaneously
	size_t freeSlotIndex;
	if (_freeSlotIndices[gpuId].size() > 0) {
		// Reuse free slot in pool
		freeSlotIndex = _freeSlotIndices[gpuId].back();
		_freeSlotIndices[gpuId].pop_back();
	} else {
		freeSlotIndex = _slots.size();
		_slots.emplace_back();
	}

	_currentAssignations[task] = freeSlotIndex;

	return freeSlotIndex;
}

void CUDAReductionStorage::releaseSlotsInUse(Task *task, ComputePlace *computePlace)
{
	std::lock_guard<ReductionInfo::spinlock_t> guard(_lock);

	assert(computePlace->getType() == nanos6_cuda_device);
	assignation_map_t::iterator itSlot = _currentAssignations.find(task);

	if (itSlot != _currentAssignations.end()) {
		long int currentSlotIndex = itSlot->second;
		int gpuId = computePlace->getIndex();
		assert((size_t)gpuId < _freeSlotIndices.size());

		assert(_slots[currentSlotIndex].storage != nullptr);
		assert(_slots[currentSlotIndex].initialized);
		assert(_slots[currentSlotIndex].deviceId == gpuId);
		_freeSlotIndices[gpuId].emplace_back(currentSlotIndex);
		_currentAssignations[task] = -1;
	}
}
