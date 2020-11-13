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
	static EnvironmentVariable<bool> _NUMATrackingEnabled("NANOS6_NUMA_TRACKING", 0);
	static EnvironmentVariable<std::string> _NUMATrackingType("NANOS6_NUMA_TRACKING_TYPE", "DEPS");
	static EnvironmentVariable<bool> _NUMASchedulingEnabled("NANOS6_NUMA_SCHEDULING", 0);
	static EnvironmentVariable<bool> _NUMAStealingEnabled("NANOS6_NUMA_STEALING", 0);

	static const double RW_BONUS_FACTOR = 2.0;
	static const uint64_t DISTANCE_THRESHOLD = 15;
	static const uint64_t LOAD_THRESHOLD = 20;


	static inline bool isNUMATrackingEnabled()
	{
		return _NUMATrackingEnabled;
	}

	static inline std::string getNUMATrackingType()
	{
		return _NUMATrackingType;
	}

	static inline bool isNUMASchedulingEnabled()
	{
		return _NUMASchedulingEnabled;
	}

	static inline bool isNUMAStealingEnabled()
	{
		return _NUMAStealingEnabled;
	}
}

#endif // DATA_TRACKING_SUPPORT_HPP
