/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef THREAD_HARDWARE_COUNTERS_HPP
#define THREAD_HARDWARE_COUNTERS_HPP

#include "ThreadHardwareCountersInterface.hpp"


class ThreadHardwareCounters {

private:

	//! Thread-related hardware counters for the PAPI backend
	ThreadHardwareCountersInterface *_papiCounters;

	//! Thread-related hardware counters for the PQoS backend
	ThreadHardwareCountersInterface *_pqosCounters;

public:

	inline ThreadHardwareCounters()
	{
		_papiCounters = nullptr;
		_pqosCounters = nullptr;
	}

	//! \brief Initialize and construct all backend objects
	void initialize();

	//! \brief Destroy all backend objects
	void shutdown();

	//! \brief Return the PAPI counters of the thread (if it is enabled) or nullptr
	inline ThreadHardwareCountersInterface *getPAPICounters() const
	{
		return _papiCounters;
	}

	//! \brief Return the PQOS counters of the thread (if it is enabled) or nullptr
	inline ThreadHardwareCountersInterface *getPQoSCounters() const
	{
		return _pqosCounters;
	}

};

#endif // THREAD_HARDWARE_COUNTERS_HPP
