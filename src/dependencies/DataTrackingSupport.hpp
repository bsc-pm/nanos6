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
	static EnvironmentVariable<bool> _trackingEnabled("NANOS6_DATA_TRACKING", 0);
	static EnvironmentVariable<bool> _NUMATrackingEnabled("NANOS6_NUMA_TRACKING", 0);
	static EnvironmentVariable<bool> _NUMASchedulingEnabled("NANOS6_NUMA_SCHEDULING", 0);
	static EnvironmentVariable<bool> _NUMAStealingEnabled("NANOS6_NUMA_STEALING", 0);
	static EnvironmentVariable<std::string> _NUMATrackingType("NANOS6_NUMA_TRACKING_TYPE", "DEPS");
	static EnvironmentVariable<bool> _trackingReportEnabled("NANOS6_TRACKING_REPORT", 0);
	static EnvironmentVariable<bool> _checkExpiration("NANOS6_CHECK_EXPIRATION", 0);

	//! Data tracking
	typedef int16_t location_t;
	static const int UNKNOWN_LOCATION = -1;

	typedef uint64_t timestamp_t;
	static const int NOT_PRESENT = 0;

	static const size_t MAX_TRACKING_THRESHOLD = 1024*1024*128;
	static const size_t MIN_TRACKING_THRESHOLD = 1024;

	static const double L2_THRESHOLD = 0.25;
	static const double L3_THRESHOLD = 0.70;

	static const double RW_BONUS_FACTOR = 2.0;

	enum HardwareCacheLevel {
		L2_LEVEL = 2,
		L3_LEVEL
	};

	struct DataTrackingInfo {
		timestamp_t _timeL2;
		timestamp_t _timeL3;
		location_t _location;

		DataTrackingInfo()
			: _timeL2(NOT_PRESENT), _timeL3(NOT_PRESENT), _location(UNKNOWN_LOCATION)
		{}

		DataTrackingInfo(location_t location, timestamp_t timeL2, timestamp_t timeL3)
			: _timeL2(timeL2), _timeL3(timeL3), _location(location)
		{}

		void setInfo(location_t location, timestamp_t timeL2, timestamp_t timeL3)
		{
			_location = location;
			_timeL2 = timeL2;
			_timeL3 = timeL3;
		}
	};

	static inline bool isTrackingEnabled()
	{
		return _trackingEnabled;
	}

	static inline bool isTrackingReportEnabled()
	{
		return _trackingReportEnabled;
	}

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

	static inline std::string getNUMATrackingType()
	{
		return _NUMATrackingType;
	}

	static inline bool isCheckExpirationEnabled()
	{
		return _checkExpiration;
	}
}

#endif // DATA_TRACKING_SUPPORT_HPP
