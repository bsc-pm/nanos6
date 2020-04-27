/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_HARDWARE_COUNTERS_HPP
#define TASK_HARDWARE_COUNTERS_HPP

#include "SupportedHardwareCounters.hpp"
#include "TaskHardwareCountersInterface.hpp"

#if HAVE_PAPI
#include "hardware-counters/papi/PAPITaskHardwareCounters.hpp"
#endif

#if HAVE_PQOS
#include "hardware-counters/pqos/PQoSTaskHardwareCounters.hpp"
#endif


class TaskHardwareCounters {

private:

#if HAVE_PAPI
	//! Task-related hardware counters for the PAPI backend
	PAPITaskHardwareCounters *_papiCounters;
#endif

#if HAVE_PQOS
	//! Task-related hardware counters for the PQoS backend
	PQoSTaskHardwareCounters *_pqosCounters;
#endif

	//! The base allocation address used to construct all the previous objects
	void *_allocationAddress;

public:

	inline TaskHardwareCounters() :
#if HAVE_PAPI
		_papiCounters(nullptr),
#endif
#if HAVE_PQOS
		_pqosCounters(nullptr),
#endif
		_allocationAddress(nullptr)
	{
	}

	//! \brief Initialize and construct all backend objects with the previously allocated space
	void initialize();

	//! \brief Set the allocation address for all the backend objects
	inline void setAllocationAddress(void *address)
	{
		_allocationAddress = address;
	}

	//! \brief Retreive the allocation address for all the backend objects
	inline void *getAllocationAddress() const
	{
		return _allocationAddress;
	}

	//! \brief Get the size needed to construct all the structures for all backends
	static size_t getTaskHardwareCountersSize();

	//! \brief Return the PAPI counters of the task (if it is enabled) or nullptr
	inline TaskHardwareCountersInterface *getPAPICounters() const
	{
#if HAVE_PAPI
		return (TaskHardwareCountersInterface *) _papiCounters;
#else
		return nullptr;
#endif
	}

	//! \brief Return the PQoS counters of the task (if it is enabled) or nullptr
	inline TaskHardwareCountersInterface *getPQoSCounters() const
	{
#if HAVE_PQOS
		return (TaskHardwareCountersInterface *) _pqosCounters;
#else
		return nullptr;
#endif
	}

	//! \brief Empty hardware counter structures
	inline void clear()
	{
#if HAVE_PAPI
		if (_papiCounters != nullptr) {
			_papiCounters->clear();
		}
#endif

#if HAVE_PQOS
		if (_pqosCounters != nullptr) {
			_pqosCounters->clear();
		}
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

	//! \brief Get the accumulated value of a HW counter
	//!
	//! \param[in] counterId The type of counter to get the accumulation from
	inline double getAccumulated(HWCounters::counters_t counterId)
	{
		if (counterId >= HWCounters::PQOS_MIN_EVENT && counterId <= HWCounters::PQOS_MAX_EVENT) {
#if HAVE_PQOS
			assert(_pqosCounters != nullptr);
			return _pqosCounters->getAccumulated(counterId);
#else
			assert(false);
#endif
		} else if (counterId >= HWCounters::PAPI_MIN_EVENT && counterId <= HWCounters::PAPI_MAX_EVENT) {
#if HAVE_PAPI
			assert(_papiCounters != nullptr);
			return _papiCounters->getAccumulated(counterId);
#else
			assert(false);
#endif
		} else {
			assert(false);
		}

		return 0.0;
	}

};

#endif // TASK_HARDWARE_COUNTERS_HPP
