/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/task-instantiation.h>

#include "HardwareInfo.hpp"
#include "hwinfo/HostInfo.hpp"

#include <config.h>

#ifdef USE_CUDA
#include "cuda/CUDAInfo.hpp"
#endif //USE_CUDA

std::vector<DeviceInfo *> HardwareInfo::_infos;

void HardwareInfo::initialize()
{
	_infos.resize(nanos6_device_t::nanos6_device_type_num);
	
	_infos[nanos6_device_t::nanos6_host_device] = new HostInfo();
	
#ifdef USE_CUDA
	_infos[nanos6_device_t::nanos6_cuda_device] = new CUDAInfo();
#endif //USE_CUDA
	
	for (int i = 0; i < nanos6_device_t::nanos6_device_type_num; ++i) {
		if (_infos[i] != nullptr) {
			_infos[i]->initialize();
		}
	}
}

void HardwareInfo::shutdown()
{
	for (int i = 0; i < nanos6_device_t::nanos6_device_type_num; ++i) {
		if (_infos[i] != nullptr) {
			_infos[i]->shutdown();
		}
	}
}
