/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <config.h>
#include <nanos6/task-instantiation.h>

#include "HardwareInfo.hpp"
#include "hwinfo/HostInfo.hpp"

#ifdef USE_CUDA
#include "hardware/device/cuda/CUDADeviceInfo.hpp"
#endif

std::vector<DeviceInfo *> HardwareInfo::_infos;
thread_local Task* HardwareInfo::threadTask;

void HardwareInfo::initialize()
{
	_infos.resize(nanos6_device_t::nanos6_device_type_num);

	_infos[nanos6_device_t::nanos6_host_device] = new HostInfo();
#ifdef USE_CUDA
	_infos[nanos6_device_t::nanos6_cuda_device] = new CUDADeviceInfo();
#endif
// Fill the rest of the devices accordingly, once implemented
}

void HardwareInfo::shutdown()
{
	for (int i = 0; i < nanos6_device_t::nanos6_device_type_num; ++i) {
		if (_infos[i] != nullptr) {
			delete _infos[i];
		}
	}
}

