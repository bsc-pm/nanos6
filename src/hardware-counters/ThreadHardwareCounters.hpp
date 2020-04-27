/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef THREAD_HARDWARE_COUNTERS_HPP
#define THREAD_HARDWARE_COUNTERS_HPP

#include "ThreadHardwareCountersInterface.hpp"

#if HAVE_PAPI
#include "hardware-counters/papi/PAPIThreadHardwareCounters.hpp"
#endif

#if HAVE_PQOS
#include "hardware-counters/pqos/PQoSThreadHardwareCounters.hpp"
#endif


class ThreadHardwareCounters {

private:

#if HAVE_PAPI
	//! Thread-related hardware counters for the PAPI backend
	PAPIThreadHardwareCounters *_papiCounters;
#endif

#if HAVE_PQOS
	//! Thread-related hardware counters for the PQoS backend
	PQoSThreadHardwareCounters *_pqosCounters;
#endif

public:

	inline ThreadHardwareCounters()
	{
#if HAVE_PAPI
		_papiCounters = nullptr;
#endif
#if HAVE_PQOS
		_pqosCounters = nullptr;
#endif
	}

	//! \brief Initialize and construct all backend objects
	void initialize();

	//! \brief Return the PAPI counters of the thread (if it is enabled) or nullptr
	inline ThreadHardwareCountersInterface *getPAPICounters() const
	{
#if HAVE_PAPI
		return _papiCounters;
#else
		return nullptr;
#endif
	}

	//! \brief Return the PQOS counters of the thread (if it is enabled) or nullptr
	inline ThreadHardwareCountersInterface *getPQoSCounters() const
	{
#if HAVE_PQOS
		return _pqosCounters;
#else
		return nullptr;
#endif
	}

};

#endif // THREAD_HARDWARE_COUNTERS_HPP
