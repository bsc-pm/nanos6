/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_TRACKING_SUPPORT_HPP
#define DATA_TRACKING_SUPPORT_HPP

#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/SpinLock.hpp"

class Task;

namespace DataTrackingSupport {
	static EnvironmentVariable<bool> _NUMATrackingEnabled("NANOS6_NUMA_TRACKING", 1);
	static EnvironmentVariable<bool> _NUMASchedulingEnabled("NANOS6_NUMA_SCHEDULING", 1);
	static EnvironmentVariable<bool> _NUMAStealingEnabled("NANOS6_NUMA_STEALING", 1);

	static const double RW_BONUS_FACTOR = 2.0;
	static const uint64_t DISTANCE_THRESHOLD = 15;
	static const uint64_t LOAD_THRESHOLD = 20;
	extern uint64_t IS_THRESHOLD;

	typedef int16_t location_t;

	enum HardwareCacheLevel {
		L2_LEVEL = 2,
		L3_LEVEL
	};

	static inline bool isNUMATrackingEnabled()
	{
		return _NUMATrackingEnabled;
	}

	static inline bool isNUMASchedulingEnabled()
	{
		return _NUMASchedulingEnabled;
	}

	static inline bool isNUMAStealingEnabled()
	{
		return _NUMAStealingEnabled;
	}

	extern size_t getNUMATrackingNodes();

	extern void setISThreshold(uint64_t threshold);
}

#endif // DATA_TRACKING_SUPPORT_HPP
