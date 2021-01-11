/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2021 Barcelona Supercomputing Center (BSC)
*/

#include <config.h>
#include <nanos6/task-instantiation.h>

#include "HardwareInfo.hpp"
#include "hwinfo/HostInfo.hpp"

#ifdef USE_CUDA
#include "hardware/device/cuda/CUDADeviceInfo.hpp"
#endif

#ifdef USE_OPENACC
#include "hardware/device/openacc/OpenAccDeviceInfo.hpp"
#endif

std::vector<DeviceInfo *> HardwareInfo::_infos;

void HardwareInfo::initialize()
{
	_infos.resize(nanos6_device_type_num);

	_infos[nanos6_host_device] = new HostInfo();
	// Prioritizing OpenACC over CUDA, as CUDA calls appear to break PGI contexts.
	// The opposite fortunately does not appear to happen.
#ifdef USE_OPENACC
	_infos[nanos6_openacc_device] = new OpenAccDeviceInfo();
#endif
#ifdef USE_CUDA
	_infos[nanos6_cuda_device] = new CUDADeviceInfo();
#endif
// Fill the rest of the devices accordingly, once implemented
}

void HardwareInfo::initializeDeviceServices()
{
	_infos[nanos6_host_device]->initializeDeviceServices();
#ifdef USE_OPENACC
	_infos[nanos6_openacc_device]->initializeDeviceServices();
#endif
#ifdef USE_CUDA
	_infos[nanos6_cuda_device]->initializeDeviceServices();
#endif
}

void HardwareInfo::shutdown()
{
	for (int i = 0; i < nanos6_device_type_num; ++i) {
		if (_infos[i] != nullptr) {
			delete _infos[i];
		}
	}
}

void HardwareInfo::shutdownDeviceServices()
{
	_infos[nanos6_host_device]->shutdownDeviceServices();
#ifdef USE_OPENACC
	_infos[nanos6_openacc_device]->shutdownDeviceServices();
#endif
#ifdef USE_CUDA
	_infos[nanos6_cuda_device]->shutdownDeviceServices();
#endif
}
