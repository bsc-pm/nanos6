/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_PLACE_HPP
#define CPU_PLACE_HPP

#include <atomic>

#include "ComputePlace.hpp"
#include "L2Cache.hpp"
#include "L3Cache.hpp"

class CPUPlace : public ComputePlace {
private:
	L2Cache *_l2Cache;
	L3Cache *_l3Cache;

public:
	CPUPlace(int index, bool owned = true) :
		ComputePlace(index, nanos6_device_t::nanos6_host_device, owned),
		_l2Cache(nullptr),
		_l3Cache(nullptr)
	{
	}

	CPUPlace (int index, L2Cache *l2Cache, L3Cache *l3Cache, bool owned = true) :
		ComputePlace(index, nanos6_device_t::nanos6_host_device, owned),
		_l2Cache(l2Cache),
		_l3Cache(l3Cache)
	{
	}

	~CPUPlace()
	{
	}

	inline bool hasL3Cache() const
	{
		return (_l3Cache != nullptr);
	}

	inline void setL2Cache(L2Cache *l2Cache)
	{
		_l2Cache = l2Cache;
	}

	inline L2Cache *getL2Cache() const
	{
		return _l2Cache;
	}

	inline void setL3Cache(L3Cache *l3Cache)
	{
		_l3Cache = l3Cache;
	}

	inline L3Cache *getL3Cache() const
	{
		return _l3Cache;
	}
};

#endif // CPU_PLACE_HPP
