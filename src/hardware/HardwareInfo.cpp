/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/task-instantiation.h>

#include "HardwareInfo.hpp"
#include "device/DeviceInfoImplementation.hpp"
#include "device/implementation/CUDA.hpp"
#include "device/implementation/FPGA.hpp"
#include "hwinfo/HostInfo.hpp"

std::vector<DeviceInfo *> HardwareInfo::_infos;
std::vector<DeviceFunctionsInterface *> HardwareInfo::_functions;

void HardwareInfo::initialize()
{
	_infos.resize(nanos6_device_t::nanos6_device_type_num);
	_functions.resize(nanos6_device_t::nanos6_device_type_num);
	
	_infos[nanos6_device_t::nanos6_host_device] = new HostInfo();
	
	DeviceFunctionsInterface* cuda_functions = new CUDAFunctions();
	cuda_functions->initialize();
	_functions[nanos6_cuda_device] = cuda_functions;
	_infos[nanos6_cuda_device] = new DeviceInfoImplementation(nanos6_cuda_device, cuda_functions);
	
	DeviceFunctionsInterface* fpga_functions = new FPGAFunctions();
	fpga_functions->initialize();
	_functions[nanos6_fpga_device] = fpga_functions;
	_infos[nanos6_fpga_device] = new DeviceInfoImplementation(nanos6_fpga_device,fpga_functions);
	
	for (int i = 0; i < nanos6_device_t::nanos6_device_type_num; ++i) {
		if (_infos[i] != nullptr) {
			_infos[i]->initialize();
		}
	}
}

void HardwareInfo::shutdown()
{
	for (int i = 0; i < nanos6_device_t::nanos6_device_type_num; ++i) {
		if (_functions[i] != nullptr) {
			_functions[i]->shutdown();
			delete _functions[i];
		}
		if (_infos[i] != nullptr) {
			_infos[i]->shutdown();
			delete _infos[i];
		}
	}
}
	
bool HardwareInfo::canDeviceRunTasks(nanos6_device_t type)
{
	if (type == nanos6_host_device) {
		return true;
	}
	return _functions[type]->getInitStatus();
}
