/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_TRACKING_SUPPORT_HPP
#define DATA_TRACKING_SUPPORT_HPP

#include "hardware/HardwareInfo.hpp"
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

	static inline size_t getNUMATrackingNodes()
	{
        if (_NUMATrackingEnabled) {
            return HardwareInfo::getValidMemoryPlaceCount(nanos6_host_device);
        } else {
            return 1;
        }
	}
}

#endif // DATA_TRACKING_SUPPORT_HPP
