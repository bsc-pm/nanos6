/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef HARDWARE_CACHE_HPP
#define HARDWARE_CACHE_HPP

#include <atomic>
#include <cassert>
#include <vector>

class CPU;

class HardwareCache {
public:
	enum HardwareCacheLevel {
		L2_LEVEL = 2,
		L3_LEVEL
	};

protected:
	int _id;
	size_t _cacheSize;
	size_t _cacheLineSize;
	HardwareCacheLevel _cacheLevel;
	//! CPUs that access this cache.
	std::vector<CPU *> _cpus;

public:
	HardwareCache(int id, size_t cacheSize, size_t cacheLineSize, HardwareCacheLevel cacheLevel) :
		_id(id),
		_cacheSize(cacheSize),
		_cacheLineSize(cacheLineSize),
		_cacheLevel(cacheLevel)
	{
	}

	virtual ~HardwareCache()
	{
	}

	inline void setId(int id)
	{
		_id = id;
	}

	inline int getId() const
	{
		return _id;
	}

	inline size_t getCacheSize() const
	{
		return _cacheSize;
	}

	inline size_t getCacheLineSize() const
	{
		return _cacheLineSize;
	}

	inline HardwareCacheLevel getCacheLevel() const
	{
		return _cacheLevel;
	}

	inline size_t getNumCPUs() const
	{
		return _cpus.size();
	}

	inline void addCPU(CPU * cpu)
	{
		_cpus.push_back(cpu);
	}

	inline void clearCPUs()
	{
		_cpus.clear();
	}

	inline std::vector<CPU *> const &getCPUs()
	{
		return _cpus;
	}
};

#endif // HARDWARE_CACHE_HPP
