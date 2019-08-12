/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#include <vector>

#include "Device.hpp"
#include "hardware/places/DeviceComputePlace.hpp"

void Device::addComputePlace(DeviceComputePlace *computePlace)
{
	_places.push_back(computePlace);
}

Device::Device(nanos6_device_t type, int subType) :
		_devType(type), _subType(subType)
{
}
