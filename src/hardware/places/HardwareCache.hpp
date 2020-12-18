/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef HARDWARE_CACHE_HPP
#define HARDWARE_CACHE_HPP

#include <atomic>
#include <cassert>
#include <vector>

#include "dependencies/DataTrackingSupport.hpp"

class CPU;
class L3Cache;

class HardwareCache {
protected:
	unsigned _id;
	size_t _cacheSize;
	size_t _cacheLineSize;
	size_t _cacheLevel;
	//! CPUs that access this cache.
	std::vector<CPU *> _cpus;

public:
	HardwareCache (unsigned id, size_t cacheSize, size_t cacheLineSize, DataTrackingSupport::HardwareCacheLevel cacheLevel)
		: _id(id), _cacheSize(cacheSize), _cacheLineSize(cacheLineSize), _cacheLevel(cacheLevel)
	{}

	virtual ~HardwareCache() {}

	virtual inline void setId(unsigned id)
	{
		_id = id;
	}

	virtual inline unsigned getId() const
	{
		return _id;
	}

	virtual inline size_t getCacheSize() const
	{
		return _cacheSize;
	}

	virtual inline size_t getCacheLineSize() const
	{
		return _cacheLineSize;
	}

	virtual inline size_t getCacheLevel() const
	{
		return _cacheLevel;
	}

	virtual inline size_t getCPUNumber()
	{
		return _cpus.size();
	}

	virtual inline void addCPU(CPU * cpu)
	{
		_cpus.push_back(cpu);
	}

	virtual inline void clearCPUs()
	{
		_cpus.clear();
	}

	virtual inline std::vector<CPU *> const &getCPUs()
	{
		return _cpus;
	}
};

#endif // HARDWARE_CACHE_HPP
