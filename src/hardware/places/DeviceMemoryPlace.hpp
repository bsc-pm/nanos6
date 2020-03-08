/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEVICE_MEMORY_PLACE_HPP
#define DEVICE_MEMORY_PLACE_HPP

#include "MemoryPlace.hpp"

class DeviceMemoryPlace: public MemoryPlace {
public:
	DeviceMemoryPlace(int index, nanos6_device_t type)
		: MemoryPlace(index, type)
	{
	}
	
	~DeviceMemoryPlace()
	{
	}
};

#endif // DEVICE_MEMORY_PLACE_HPP
