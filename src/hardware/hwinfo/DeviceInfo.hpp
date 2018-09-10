/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEVICE_INFO_HPP
#define DEVICE_INFO_HPP

#include "hardware/places/MemoryPlace.hpp"
#include "hardware/places/ComputePlace.hpp"

class DeviceInfo {
public:
	virtual void initialize() = 0;
	virtual void shutdown() = 0;
	
	virtual size_t getComputePlaceCount(void) = 0;
	virtual ComputePlace* getComputePlace(int index) = 0;
	
	virtual size_t getMemoryPlaceCount(void) = 0;
	virtual MemoryPlace* getMemoryPlace(int index) = 0;
};

#endif //DEVICE_INFO_HPP
