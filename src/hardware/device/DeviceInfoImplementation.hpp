/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEVICE_INFO_IMPLEMENTATION_HPP
#define DEVICE_INFO_IMPLEMENTATION_HPP

#include "Device.hpp"
#include "hardware/HardwareInfo.hpp"
#include "hardware/device/DeviceFunctionsInterface.hpp"
#include "hardware/hwinfo/DeviceInfo.hpp"


class DeviceInfoImplementation : public DeviceInfo {
public:
	std::vector<Device *> _devices;
	DeviceFunctionsInterface *_functions;
	nanos6_device_t _device_type;
	
	DeviceInfoImplementation(nanos6_device_t type, DeviceFunctionsInterface *functions) :
			_functions(functions), _device_type(type)
	{
		_functions->getDevices(_devices);
	}
	
	Device *getDevice()
	{
		return _devices[0];
	}
	
	Device *getDevice(int subtype)
	{
		for (Device* dev : _devices) {
			if (dev->getDeviceSubType() == subtype)
				return dev;
		}
		return nullptr;
	}
	
	int getDeviceSubType(int idx)
	{
		return _devices[idx]->getDeviceSubType();
	}
	
	nanos6_device_t getDeviceType(int idx)
	{
		return _devices[idx]->getDeviceType();
	}
	
	int getNumDevices()
	{
		return _devices.size();
	}
	
	Device *getDeviceNum(int idx)
	{
		return _devices[idx];
	}
	
	void initialize()
	{
		
	}
	
	void shutdown()
	{
		for (Device *dev : _devices) {
			delete dev;
		}
	}
	
	size_t getComputePlaceCount()
	{
		FatalErrorHandler::failIf(true, "This method can't be called for a DeviceInfo");
		return 0;
	}
	
	ComputePlace *getComputePlace(int)
	{
		FatalErrorHandler::failIf(true, "This method can't be called for a DeviceInfo");
		return nullptr;
	}
	
	size_t getMemoryPlaceCount()
	{
		FatalErrorHandler::failIf(true, "This method can't be called for a DeviceInfo");
		return 0;
	}
	
	MemoryPlace* getMemoryPlace(int)
	{
		FatalErrorHandler::failIf(true, "This method can't be called for a DeviceInfo");
		return nullptr;
	}
	
};

#endif // DEVICE_INFO_IMPLEMENTATION_HPP 
