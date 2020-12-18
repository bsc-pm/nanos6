/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_TRACKING_SUPPORT_HPP
#define DATA_TRACKING_SUPPORT_HPP

#include "support/config/ConfigVariable.hpp"

class Task;

class DataTrackingSupport {
	//! This is a developer option, it is not in the config file, and it
	//! may be used for debugging purposes or performance evaluation
	static ConfigVariable<bool> _NUMASchedulingEnabled;

	static const double _rwBonusFactor;
	static const uint64_t _distanceThreshold;
	static const uint64_t _loadThreshold;
	static uint64_t _shouldEnableIS;

public:
	static inline bool isNUMASchedulingEnabled()
	{
		return _NUMASchedulingEnabled;
	}

	static inline double getRWBonusFactor()
	{
		return _rwBonusFactor;
	}

	static inline double getDistanceThreshold()
	{
		return _distanceThreshold;
	}

	static inline double getLoadThreshold()
	{
		return _loadThreshold;
	}

	static bool shouldEnableIS(Task *task);

	static inline void setShouldEnableIS(uint64_t ISThreshold)
	{
		_shouldEnableIS = ISThreshold;
	}
};

#endif // DATA_TRACKING_SUPPORT_HPP
