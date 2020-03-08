/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEVICEFUNCTIONSINTERFACE_HPP
#define DEVICEFUNCTIONSINTERFACE_HPP

#include <vector>

#include <api/nanos6.h>

#include "lowlevel/FatalErrorHandler.hpp"
#include "tasks/Task.hpp"


class Device;

class DeviceFunctionsInterface {
	
protected:
	std::vector<Device *> _devices; 
public:
	
	enum deviceMemcpy {
		DEVICE_TO_HOST,
		HOST_TO_DEVICE, 
		DEVICE_TO_DEVICE
	};
	
	enum deviceError {
		Success = 0,
		ErrorMissingConfiguration,
		ErrorMemoryAllocation,
		ErrorLaunchFailure = 4
	};
	
	virtual ~DeviceFunctionsInterface()
	{
	}
	
	//returns the type associated to the function object
	virtual nanos6_device_t getType() = 0;
	
	//a malloc function that reserves memory on the device, it follows the CUDA approach
	//you pass a pointer to a pointer and a size to be allocated. The modified pointer should
	//only be used as a handler, as it's the device pointer, never access directly.
	virtual int malloc(void **ptr, size_t size) = 0;
	
	//a free function that frees a previous malloc for the device.
	virtual void free(void *ptr) = 0;
	
	//a mem copy function that should be able to transfer data:
	//device_to_device, host_to_device and device_to_host.
	virtual int memcpy(void *dst, void *src, size_t size, deviceMemcpy type) = 0;
	
	//You set the device to be used in a multi-device configuration. (Example: CUDA Index)
	virtual int setDevice(int device) = 0;
	
	//Gets the name of the function type.
	virtual const char *getName() = 0;
	
	//Generates the extra data that a task  need to be executed.
	virtual void *generateDeviceExtra(Task *task, void *extra = nullptr) = 0;
	
	//A function that is executed if the task doesn't has a body
	virtual void bodyDevice(Task *task, void *extra = nullptr) = 0;
	
	//A function that is executed after the task launched.
	virtual void postBodyDevice(Task *task, void *extra = nullptr) = 0;
	
	//A function that must check the finished tasks.
	virtual void getFinishedTasks(std::vector<Task *> &finished_tasks) = 0;
	
	//A function that must return the set of Devics of the selected type.
	virtual void getDevices(std::vector<Device *> &ret) = 0;
	
	//A function to be executed at program exit
	virtual void shutdown() = 0;
	
	//A function that prefetches the device unified memory
	virtual void unifiedAsyncPrefetch(void *pHost, size_t size, int dstDevice) = 0;
	
	//A function that registers a host region as unified memory with the device
	virtual void unifiedMemRegister(void *pHost, size_t size) = 0;
	
	//A function that unregisters a host region, freeing it from the device.
	virtual void unifiedMemUnregister(void *pHost) = 0;
	
	virtual void *unifiedGetDevicePointer(void *pHost) = 0;
	
	//returns true if initialization was correct
	virtual bool initialize() = 0;

	//returns if the initialization was correct
	virtual bool getInitStatus() = 0;

	/*external device-dependent api calls*/
	virtual void fpgaAddArg(int, uint64_t, void *, void *, char)
	{
		FatalErrorHandler::warnIf(true,"not implemented for this device");
	}
	
};

#endif /*DEVICEFUNCTIONSINTERFACE_HPP*/
