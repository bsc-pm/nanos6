/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/


#include <cassert>
#include <sys/mman.h>

#include <nanos6/reductions.h>
#include <config.h>

#include "ReductionInfo.hpp"
#include "DeviceReductionStorage.hpp"
#include "devices/HostReductionStorage.hpp"
#include "devices/CUDAReductionStorage.hpp"

#include <hardware/HardwareInfo.hpp>
#include <MemoryAllocator.hpp>
#include <executors/threads/WorkerThread.hpp>

ReductionInfo::ReductionInfo(void *address, size_t length, reduction_type_and_operator_index_t typeAndOperatorIndex,
	std::function<void(void *, void *, size_t)> initializationFunction, std::function<void(void *, void *, size_t)> combinationFunction) :
	_address(address),
	_length(length),
	_paddedLength(((length + HardwareInfo::getCacheLineSize() - 1) / HardwareInfo::getCacheLineSize()) * HardwareInfo::getCacheLineSize()),
	_typeAndOperatorIndex(typeAndOperatorIndex),
	_initializationFunction(std::bind(initializationFunction, std::placeholders::_1, address, std::placeholders::_2)),
	_combinationFunction(combinationFunction),
	_registeredAccesses(2)
{
}

ReductionInfo::~ReductionInfo()
{
	assert(_registeredAccesses == 0);

	for(std::pair<nanos6_device_t, DeviceReductionStorage *> device : _deviceStorages) {
		assert(device.second != nullptr);
		delete device.second;
	}
}

reduction_type_and_operator_index_t ReductionInfo::getTypeAndOperatorIndex() const
{
	return _typeAndOperatorIndex;
}

const void *ReductionInfo::getOriginalAddress() const
{
	return _address;
}

size_t ReductionInfo::getOriginalLength() const
{
	return _length;
}

void ReductionInfo::combine()
{
	std::lock_guard<spinlock_t> guard(_lock);
	assert(_address != nullptr);

	char *originalAddress = (char*)_address;

	for(std::pair<nanos6_device_t, DeviceReductionStorage *> device : _deviceStorages) {
		assert(device.second != nullptr);
		device.second->combineInStorage(originalAddress);
	}
}

void ReductionInfo::releaseSlotsInUse(Task* task, ComputePlace* computePlace) {
	std::lock_guard<spinlock_t> guard(_lock);
	nanos6_device_t deviceType = computePlace->getType();

	DeviceReductionStorage * storage = _deviceStorages[deviceType];
	if(storage == nullptr)
		return;

	storage->releaseSlotsInUse(task, computePlace);
}

DeviceReductionStorage * ReductionInfo::allocateDeviceStorage(nanos6_device_t deviceType) {
	assert(_deviceStorages[deviceType] == nullptr);
	DeviceReductionStorage * storage = nullptr;

	switch(deviceType) {
		case nanos6_device_t::nanos6_host_device:
			storage = new HostReductionStorage(_address, _length, _paddedLength,
				_initializationFunction, _combinationFunction);
			break;
#if USE_CUDA
		case nanos6_device_t::nanos6_cuda_device:
			storage = new CUDAReductionStorage(_address, _length, _paddedLength,
				_initializationFunction, _combinationFunction);
			break;
#endif
		default:
			break;
	}

	assert(storage != nullptr);
	_deviceStorages[deviceType] = storage;
	return storage;
}

void * ReductionInfo::getFreeSlot(Task* task, ComputePlace* computePlace) {
	std::lock_guard<spinlock_t> guard(_lock);

	nanos6_device_t deviceType = computePlace->getType();

	DeviceReductionStorage * storage = _deviceStorages[deviceType];
	if(storage == nullptr)
		storage = allocateDeviceStorage(deviceType);

	assert(storage != nullptr);

	size_t index = storage->getFreeSlotIndex(task, computePlace);

	void * privateStorage = storage->getFreeSlotStorage(task, index, computePlace);
	assert(privateStorage != nullptr);

	return privateStorage;
}
