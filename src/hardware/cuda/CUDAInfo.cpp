/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "CUDAInfo.hpp"

void CUDAInfo::initialize()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	
	if (deviceCount == 0) {
		return;	
	}
	
	_devices.resize(deviceCount);
	_pollingServices.resize(deviceCount);
    	for (int i = 0; i < deviceCount; ++i) {
		setDevice(i);	
			
		_devices[i] = new CUDADevice(i);
		
		_pollingServices[i] = new CUDAPollingService(_devices[i]);
		_pollingServices[i]->start();
	}
}

void CUDAInfo::shutdown()
{
	for (int i = 0; i < _devices.size(); ++i) {
		delete _devices[i];
		
		_pollingServices[i]->stop();
		delete _pollingServices[i];
	}
}

