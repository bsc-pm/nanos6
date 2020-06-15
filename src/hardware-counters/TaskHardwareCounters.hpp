/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_HARDWARE_COUNTERS_HPP
#define TASK_HARDWARE_COUNTERS_HPP

#include "HardwareCounters.hpp"
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

	//! The base allocation address used to construct all the previous objects
	void *_allocationAddress;

private:

	static inline size_t getPAPITaskHardwareCountersSize()
	{
#if HAVE_PAPI
		return sizeof(PAPITaskHardwareCounters);
#endif
		return 0;
	}

	static inline size_t getPQoSTaskHardwareCountersSize()
	{
#if HAVE_PQOS
		return sizeof(PQoSTaskHardwareCounters);
#endif
		return 0;
	}

public:

	inline TaskHardwareCounters(void *allocationAddress) :
		_allocationAddress(allocationAddress)
	{
	}

	//! \brief Initialize and construct all backend objects with the previously allocated space
	//!
	//! \param[in] enabled Whether hardware counters are enabled for the task
	inline void initialize(__attribute__((unused)) bool enabled)
	{
		// NOTE: Objects are constructed in this function, but they are freed when
		// the task is freed (see TaskFinalizationImplementation.hpp)

		// Use a copy since we may need the original allocation address
		__attribute__((unused)) void *currentAddress = _allocationAddress;
#if HAVE_PAPI
		if (HardwareCounters::isBackendEnabled(HWCounters::PAPI_BACKEND)) {
			assert(currentAddress != nullptr);

			new (currentAddress) PAPITaskHardwareCounters(enabled);
			currentAddress = (char *) currentAddress + sizeof(PAPITaskHardwareCounters);
		}
#endif

#if HAVE_PQOS
		if (HardwareCounters::isBackendEnabled(HWCounters::PQOS_BACKEND)) {
			assert(currentAddress != nullptr);

			new (currentAddress) PQoSTaskHardwareCounters(enabled);
			currentAddress = (char *) currentAddress + sizeof(PQoSTaskHardwareCounters);
		}
#endif
	}

	//! \brief Destroy all backend objects
	inline void shutdown()
	{
		TaskHardwareCountersInterface *papiCounters = getPAPICounters();
		if (papiCounters != nullptr) {
#if HAVE_PAPI
			((PAPITaskHardwareCounters *) papiCounters)->~PAPITaskHardwareCounters();
#endif
		}

		TaskHardwareCountersInterface *pqosCounters = getPQoSCounters();
		if (pqosCounters != nullptr) {
#if HAVE_PQOS
			((PQoSTaskHardwareCounters *) pqosCounters)->~PQoSTaskHardwareCounters();
#endif
		}
	}

	//! \brief Retreive the allocation address for all the backend objects
	inline void *getAllocationAddress() const
	{
		return _allocationAddress;
	}

	//! \brief Return the PAPI counters of the task (if it is enabled) or nullptr
	inline TaskHardwareCountersInterface *getPAPICounters() const
	{
#if HAVE_PAPI
		if (HardwareCounters::isBackendEnabled(HWCounters::PAPI_BACKEND)) {
			return (TaskHardwareCountersInterface *) _allocationAddress;
		}
#endif
		return nullptr;
	}

	//! \brief Return the PQoS counters of the task (if it is enabled) or nullptr
	inline TaskHardwareCountersInterface *getPQoSCounters() const
	{
#if HAVE_PQOS
		if (HardwareCounters::isBackendEnabled(HWCounters::PQOS_BACKEND)) {
			void *papiCounters = getPAPICounters();
			return (papiCounters == nullptr) ?
				(TaskHardwareCountersInterface *) _allocationAddress :
				(TaskHardwareCountersInterface *) ((char *) papiCounters + getPAPITaskHardwareCountersSize());
		}
#endif
		return nullptr;
	}

	//! \brief Get the size needed to construct all the structures for all backends
	static inline size_t getTaskHardwareCountersSize()
	{
		size_t totalSize = 0;

		if (HardwareCounters::isBackendEnabled(HWCounters::PAPI_BACKEND)) {
			totalSize += getPAPITaskHardwareCountersSize();
		}

		if (HardwareCounters::isBackendEnabled(HWCounters::PQOS_BACKEND)) {
			totalSize += getPQoSTaskHardwareCountersSize();
		}

		return totalSize;
	}

	//! \brief Get the delta value of a HW counter
	//!
	//! \param[in] counterType The type of counter to get the delta from
	inline uint64_t getDelta(HWCounters::counters_t counterType)
	{
		TaskHardwareCountersInterface *taskCounters = nullptr;
		if (counterType >= HWCounters::PAPI_MIN_EVENT && counterType <= HWCounters::PAPI_MAX_EVENT) {
			taskCounters = getPAPICounters();
		} else if (counterType >= HWCounters::PQOS_MIN_EVENT && counterType <= HWCounters::PQOS_MAX_EVENT) {
			taskCounters = getPQoSCounters();
		}
		assert(taskCounters != nullptr);

		return taskCounters->getDelta(counterType);
	}

	//! \brief Get the accumulated value of a HW counter
	//!
	//! \param[in] counterType The type of counter to get the accumulation from
	inline uint64_t getAccumulated(HWCounters::counters_t counterType)
	{
		TaskHardwareCountersInterface *taskCounters = nullptr;
		if (counterType >= HWCounters::PAPI_MIN_EVENT && counterType <= HWCounters::PAPI_MAX_EVENT) {
			taskCounters = getPAPICounters();
		} else if (counterType >= HWCounters::PQOS_MIN_EVENT && counterType <= HWCounters::PQOS_MAX_EVENT) {
			taskCounters = getPQoSCounters();
		}
		assert(taskCounters != nullptr);

		return taskCounters->getAccumulated(counterType);
	}

};

#endif // TASK_HARDWARE_COUNTERS_HPP
