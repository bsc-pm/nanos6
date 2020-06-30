/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PAPI_TASK_HARDWARE_COUNTERS_HPP
#define PAPI_TASK_HARDWARE_COUNTERS_HPP

#include <papi.h>
#include <string>

#include "PAPIHardwareCounters.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"
#include "hardware-counters/TaskHardwareCountersInterface.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


class PAPITaskHardwareCounters : public TaskHardwareCountersInterface {

private:

	//! Arrays of regular HW counter deltas and accumulations
	long long *_countersDelta;
	long long *_countersAccumulated;

public:

	inline PAPITaskHardwareCounters(void *allocationAddress)
	{
		assert(allocationAddress != nullptr);

		const size_t numEvents = PAPIHardwareCounters::getNumEnabledCounters();
		_countersDelta = (long long *) allocationAddress;
		_countersAccumulated = (long long *) ((char *) allocationAddress + (numEvents * sizeof(long long)));

		clear();
	}

	//! \brief Empty hardware counter structures
	inline void clear() override
	{
		const size_t numCounters = PAPIHardwareCounters::getNumEnabledCounters();
		memset(_countersDelta, 0, numCounters * sizeof(long long));
		memset(_countersAccumulated, 0, numCounters * sizeof(long long));
	}

	//! \brief Read counters from an event set
	//!
	//! \param[in] eventSet The event set specified
	inline void readCounters(int eventSet)
	{
		assert(eventSet != PAPI_NULL);

		int ret = PAPI_read(eventSet, _countersDelta);
		if (ret != PAPI_OK) {
			FatalErrorHandler::fail(ret, " when reading a PAPI event set - ", PAPI_strerror(ret));
		}

		ret = PAPI_reset(eventSet);
		if (ret != PAPI_OK) {
			FatalErrorHandler::fail(ret, " when resetting a PAPI event set - ", PAPI_strerror(ret));
		}

		const size_t numEvents = PAPIHardwareCounters::getNumEnabledCounters();
		for (size_t i = 0; i < numEvents; ++i) {
			_countersAccumulated[i] += _countersDelta[i];
		}
	}

	//! \param[in] counterType The type of counter to get the delta from
	inline uint64_t getDelta(HWCounters::counters_t counterType) override
	{
		assert(PAPIHardwareCounters::isCounterEnabled(counterType));

		int innerId = PAPIHardwareCounters::getInnerIdentifier(counterType);
		assert(innerId >= 0 && (size_t) innerId < PAPIHardwareCounters::getNumEnabledCounters());

		return (uint64_t) _countersDelta[innerId];
	}

	//! \brief Get the accumulated value of a HW counter
	//!
	//! \param[in] counterType The type of counter to get the accumulation from
	inline uint64_t getAccumulated(HWCounters::counters_t counterType) override
	{
		assert(PAPIHardwareCounters::isCounterEnabled(counterType));

		int innerId = PAPIHardwareCounters::getInnerIdentifier(counterType);
		assert(innerId >= 0 && (size_t) innerId < PAPIHardwareCounters::getNumEnabledCounters());

		return (uint64_t) _countersAccumulated[innerId];
	}

	//! \brief Combine the counters of two tasks
	//!
	//! \param[in] combineeCounters The counters of a task, which will be combined into
	//! the current counters
	inline void combineCounters(TaskHardwareCountersInterface *combineeCounters) override
	{
		PAPITaskHardwareCounters *childCounters = (PAPITaskHardwareCounters *) combineeCounters;
		assert(childCounters != nullptr);
		assert(_countersDelta != nullptr);
		assert(_countersAccumulated != nullptr);

		// Get the raw data of each regular counter and combine it
		for (size_t id = HWCounters::HWC_PAPI_MIN_EVENT; id <= HWCounters::HWC_PAPI_MAX_EVENT; ++id) {
			HWCounters::counters_t counterType = (HWCounters::counters_t) id;
			if (PAPIHardwareCounters::isCounterEnabled(counterType)) {
				int innerId = PAPIHardwareCounters::getInnerIdentifier(counterType);
				assert(innerId >= 0);

				_countersDelta[innerId] = childCounters->getDelta(counterType);
				_countersAccumulated[innerId] += childCounters->getAccumulated(counterType);
			}
		}
	}

	//! \brief Retreive the size needed for hardware counters
	static inline size_t getTaskHardwareCountersSize()
	{
		const size_t numCounters = PAPIHardwareCounters::getNumEnabledCounters();

		return numCounters * 2 * sizeof(long long);
	}

};

#endif // PAPI_TASK_HARDWARE_COUNTERS_HPP
