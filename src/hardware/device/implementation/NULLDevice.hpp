/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef NULLDEVICE_HPP
#define NULLDEVICE_HPP

#include <iostream>
#include "hardware/device/DeviceFunctionsInterface.hpp"

class NULLDevice: public DeviceFunctionsInterface {
public:
	const nanos6_device_t device_type = nanos6_device_t::nanos6_device_type_num;
	
	NULLDevice()
	{
	}
	~NULLDevice()
	{
	}
	
	int malloc(void **, size_t)
	{
		return 0;
	}
	
	void free(void *)
	{
	}
	
	int memcpy(void *, void *, size_t, deviceMemcpy)
	{
		return 0;
	}
	
	int setDevice(int)
	{
		return 0;
	}
	
	const char *getName()
	{
		return "NULLDevice";
	}
	
	void runTask(Task *, ComputePlace *)
	{
	}
	
	nanos6_device_t getType()
	{
		return nanos6_device_t::nanos6_device_type_num;
	}
	
	void *generateDeviceExtra(Task *, void *)
	{
		return nullptr;
	}
	
	void postBodyDevice(Task *, void *)
	{
	}
	
	void getFinishedTasks(std::vector<Task *>&)
	{
	}
	
	void getDevices(std::vector<Device *>&)
	{
	}
	
	void fpgaAddArg(int, uint64_t, void *, void *, char)
	{
	}
	
	void shutdown()
	{
		
	}
	
	void unifiedAsyncPrefetch(void *, size_t, int)
	{
	}
	
	void unifiedMemRegister(void *, size_t)
	{
	}
	
	void unifiedMemUnregister(void *)
	{
	}
	
	void *unifiedGetDevicePointer(void *)
	{
		return nullptr;
	}
	
	void initialize()
	{
	}
	
	void bodyDevice(Task *, void *)
	{
	}
	
};

#endif //NULLDEVICE_HPP
