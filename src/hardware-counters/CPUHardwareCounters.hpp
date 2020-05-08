/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_HARDWARE_COUNTERS_HPP
#define CPU_HARDWARE_COUNTERS_HPP

#include "CPUHardwareCountersInterface.hpp"
#include "SupportedHardwareCounters.hpp"

#if HAVE_PAPI
#include "hardware-counters/papi/PAPICPUHardwareCounters.hpp"
#endif

#if HAVE_PQOS
#include "hardware-counters/pqos/PQoSCPUHardwareCounters.hpp"
#endif


class CPUHardwareCounters {

private:

#if HAVE_PAPI
	//! CPU-related hardware counters for the PAPI backend
	PAPICPUHardwareCounters *_papiCounters;
#endif

#if HAVE_PQOS
	//! CPU-related hardware counters for the PQoS backend
	PQoSCPUHardwareCounters *_pqosCounters;
#endif

public:

	CPUHardwareCounters();

	inline ~CPUHardwareCounters()
	{
#if HAVE_PAPI
		if (_papiCounters != nullptr) {
			delete _papiCounters;
		}
#endif

#if HAVE_PQOS
		if (_pqosCounters != nullptr) {
			delete _pqosCounters;
		}
#endif
	}

	//! \brief Return the PAPI counters of the CPU (if it is enabled) or nullptr
	inline CPUHardwareCountersInterface *getPAPICounters() const
	{
#if HAVE_PAPI
		return (CPUHardwareCountersInterface *) _papiCounters;
#else
		return nullptr;
#endif
	}

	//! \brief Return the PQOS counters of the cpu (if it is enabled) or nullptr
	inline CPUHardwareCountersInterface *getPQoSCounters() const
	{
#if HAVE_PQOS
		return (CPUHardwareCountersInterface *) _pqosCounters;
#else
		return nullptr;
#endif
	}

	//! \brief Get the delta value of a HW counter
	//!
	//! \param[in] counterId The type of counter to get the delta from
	inline double getDelta(HWCounters::counters_t counterId)
	{
		if (counterId >= HWCounters::PQOS_MIN_EVENT && counterId <= HWCounters::PQOS_MAX_EVENT) {
#if HAVE_PQOS
			assert(_pqosCounters != nullptr);
			return _pqosCounters->getDelta(counterId);
#else
			assert(false);
#endif
		} else if (counterId >= HWCounters::PAPI_MIN_EVENT && counterId <= HWCounters::PAPI_MAX_EVENT) {
#if HAVE_PAPI
			assert(_papiCounters != nullptr);
			return _papiCounters->getDelta(counterId);
#else
			assert(false);
#endif
		} else {
			assert(false);
		}

		return 0.0;
	}

};

#endif // CPU_HARDWARE_COUNTERS_HPP
