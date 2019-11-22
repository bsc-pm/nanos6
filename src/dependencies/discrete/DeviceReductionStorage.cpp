/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "DeviceReductionStorage.hpp"
#include "MemoryAllocator.hpp"

DeviceReductionStorage::DeviceReductionStorage(void * address, size_t length, size_t paddedLength,
	std::function<void(void*, size_t)> initializationFunction,
	std::function<void(void*, void*, size_t)> combinationFunction) :
	_address(address),
	_length(length),
	_paddedLength(paddedLength),
	_initializationFunction(initializationFunction),
	_combinationFunction(combinationFunction)
{
}

DeviceReductionStorage::~DeviceReductionStorage()
{
}