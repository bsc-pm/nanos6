/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_HARDWARE_COUNTERS_HPP
#define TASK_HARDWARE_COUNTERS_HPP

#include "HardwareCounters.hpp"
#include "SupportedHardwareCounters.hpp"
#include "TaskHardwareCountersInterface.hpp"
#include "lowlevel/SpinLock.hpp"

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

	//! Whether monitoring of counters for this task is enabled
	bool _enabled;

	//! A spinlock to ensure a thread-safe combination of events
	SpinLock _spinlock;

public:

	inline TaskHardwareCounters(void *allocationAddress) :
		_allocationAddress(allocationAddress),
		_enabled(false),
		_spinlock()
	{
	}

	//! \brief Initialize and construct all backend objects with the previously allocated space
	//!
	//! \param[in] enabled Whether hardware counters are enabled for the task
	inline void initialize(bool enabled)
	{
		_enabled = enabled;
		if (_enabled) {
			// NOTE: Objects are constructed in this function, but they are freed when
			// the task is freed (see TaskFinalizationImplementation.hpp)

			// Use a copy since we may need the original allocation address
			__attribute__((unused)) void *currentAddress = _allocationAddress;
#if HAVE_PAPI
			if (HardwareCounters::isBackendEnabled(HWCounters::PAPI_BACKEND)) {
				assert(currentAddress != nullptr);

				// Skip sizeof(PAPITaskHardwareCounters) for the inner address
				void *innerAddress = (char *) currentAddress + sizeof(PAPITaskHardwareCounters);

				new (currentAddress) PAPITaskHardwareCounters(innerAddress);
				currentAddress = (char *) currentAddress + getPAPITaskHardwareCountersSize();
			}
#endif

#if HAVE_PQOS
			if (HardwareCounters::isBackendEnabled(HWCounters::PQOS_BACKEND)) {
				assert(currentAddress != nullptr);

				// Skip sizeof(PQoStaskHardwareCounters) for the inner address
				void *innerAddress = (char *) currentAddress + sizeof(PQoSTaskHardwareCounters);

				new (currentAddress) PQoSTaskHardwareCounters(innerAddress);
				currentAddress = (char *) currentAddress + getPQoSTaskHardwareCountersSize();
			}
#endif
		}
	}

	//! \brief Destroy all backend objects
	inline void shutdown()
	{
		if (_enabled) {
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
	}

	//! \brief Check whether hardware counter monitoring is enabled for this task
	inline bool isEnabled() const
	{
		return _enabled;
	}

	//! \brief Return the PAPI counters of the task (if it is enabled) or nullptr
	inline TaskHardwareCountersInterface *getPAPICounters() const
	{
#if HAVE_PAPI
		if (_enabled) {
			if (HardwareCounters::isBackendEnabled(HWCounters::PAPI_BACKEND)) {
				return (TaskHardwareCountersInterface *) _allocationAddress;
			}
		}
#endif
		return nullptr;
	}

	//! \brief Return the PQoS counters of the task (if it is enabled) or nullptr
	inline TaskHardwareCountersInterface *getPQoSCounters() const
	{
#if HAVE_PQOS
		if (_enabled) {
			if (HardwareCounters::isBackendEnabled(HWCounters::PQOS_BACKEND)) {
				void *papiCounters = getPAPICounters();
				return (papiCounters == nullptr) ?
					(TaskHardwareCountersInterface *) _allocationAddress :
					(TaskHardwareCountersInterface *) ((char *) papiCounters + getPAPITaskHardwareCountersSize());
			}
		}
#endif
		return nullptr;
	}

	//! \brief Get the delta value of a HW counter
	//!
	//! \param[in] counterType The type of counter to get the delta from
	inline uint64_t getDelta(HWCounters::counters_t counterType) const
	{
		if (_enabled) {
			TaskHardwareCountersInterface *taskCounters = nullptr;
			if (counterType >= HWCounters::HWC_PAPI_MIN_EVENT && counterType <= HWCounters::HWC_PAPI_MAX_EVENT) {
				taskCounters = getPAPICounters();
			} else if (counterType >= HWCounters::HWC_PQOS_MIN_EVENT && counterType <= HWCounters::HWC_PQOS_MAX_EVENT) {
				taskCounters = getPQoSCounters();
			}
			assert(taskCounters != nullptr);

			return taskCounters->getDelta(counterType);
		}

		return 0;
	}

	//! \brief Get the accumulated value of a HW counter
	//!
	//! \param[in] counterType The type of counter to get the accumulation from
	inline uint64_t getAccumulated(HWCounters::counters_t counterType) const
	{
		if (_enabled) {
			TaskHardwareCountersInterface *taskCounters = nullptr;
			if (counterType >= HWCounters::HWC_PAPI_MIN_EVENT && counterType <= HWCounters::HWC_PAPI_MAX_EVENT) {
				taskCounters = getPAPICounters();
			} else if (counterType >= HWCounters::HWC_PQOS_MIN_EVENT && counterType <= HWCounters::HWC_PQOS_MAX_EVENT) {
				taskCounters = getPQoSCounters();
			}
			assert(taskCounters != nullptr);

			return taskCounters->getAccumulated(counterType);
		}

		return 0;
	}

	//! \brief Combine the counters of two tasks
	//!
	//! \param[in] combinee The counters of a task, which will be combined into
	//! the current counters
	inline void combineCounters(const TaskHardwareCounters &combinee)
	{
		TaskHardwareCountersInterface *parentPqosCounters = getPQoSCounters();
		TaskHardwareCountersInterface *parentPapiCounters = getPAPICounters();
		TaskHardwareCountersInterface *childPqosCounters = combinee.getPQoSCounters();
		TaskHardwareCountersInterface *childPapiCounters = combinee.getPAPICounters();

		// Call each backend and let them combine their events
		_spinlock.lock();

		if (parentPqosCounters != nullptr) {
			parentPqosCounters->combineCounters(childPqosCounters);
		}

		if (parentPapiCounters != nullptr) {
			parentPapiCounters->combineCounters(childPapiCounters);
		}

		_spinlock.unlock();
	}

	//! \brief Retreive the allocation address for all the backend objects
	inline void *getAllocationAddress() const
	{
		return _allocationAddress;
	}

	//! \brief Get the size needed to construct all the structures for all backends
	static inline size_t getAllocationSize()
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

private:
	//! \brief Get the size needed to construct all the structures for PAPI
	static inline size_t getPAPITaskHardwareCountersSize()
	{
#if HAVE_PAPI
		return sizeof(PAPITaskHardwareCounters) + PAPITaskHardwareCounters::getTaskHardwareCountersSize();
#endif
		return 0;
	}

	//! \brief Get the size needed to construct all the structures for PQoS
	static inline size_t getPQoSTaskHardwareCountersSize()
	{
#if HAVE_PQOS
		return sizeof(PQoSTaskHardwareCounters) + PQoSTaskHardwareCounters::getTaskHardwareCountersSize();
#endif
		return 0;
	}
};

#endif // TASK_HARDWARE_COUNTERS_HPP
