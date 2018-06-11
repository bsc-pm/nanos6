/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "CUDAManager.hpp"

#include "lowlevel/EnvironmentVariable.hpp"

int CUDAManager::_deviceCount;
std::vector<CUDAComputePlace *> CUDAManager::_computePlaces;
std::vector<CUDAMemoryPlace *> CUDAManager::_memoryPlaces;
std::vector<CUDAHelper *> CUDAManager::_helpers;

void CUDAManager::initialize()
{
	cudaGetDeviceCount(&_deviceCount);
	if (_deviceCount == 0) {
		return;	
	}
	
	_computePlaces.resize(_deviceCount);
	_memoryPlaces.resize(_deviceCount);
	_helpers.resize(_deviceCount);
	
	EnvironmentVariable<std::string> memoryMode("NANOS6_CUDA_MEMORY", "default");
	
    for (int i = 0; i < _deviceCount; ++i) {
		cudaDeviceProp prop;
		
		cudaError_t err = cudaGetDeviceProperties(&prop, i);
		CUDAErrorHandler::handle(err, "When retrieving CUDA device properties");
		
		setDevice(i);
		
		_computePlaces[i] = new CUDAComputePlace( i, prop );
		_memoryPlaces[i] = CUDAMemoryPlace::createCUDAMemory(memoryMode.getValue(), i, prop);			
		_helpers[i] = new CUDAHelper( _computePlaces[i], _memoryPlaces[i] );
	}
	
	for (int i = 0; i < _deviceCount; ++i) {
		_helpers[i]->start();
	}
}

void CUDAManager::shutdown()
{
	for(int i = 0; i < _deviceCount; ++i){
		_helpers[i]->stop();
		delete _helpers[i];
		delete _memoryPlaces[i];
		delete _computePlaces[i];
	}
	
	_computePlaces.clear();
	_memoryPlaces.clear();
	_helpers.clear();
}

