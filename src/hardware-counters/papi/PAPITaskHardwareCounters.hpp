/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PAPI_TASK_HARDWARE_COUNTERS_HPP
#define PAPI_TASK_HARDWARE_COUNTERS_HPP

#include <string.h>

#include "PAPIHardwareCounters.hpp"
#include "hardware-counters/TaskHardwareCountersInterface.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"


class PAPITaskHardwareCounters : public TaskHardwareCountersInterface {

private:

	//! Arrays of regular HW counter deltas and accumulations
	long long *_countersDelta;

public:

	inline PAPITaskHardwareCounters(void *allocationAddress)
	{
		assert(allocationAddress != nullptr);

		_countersDelta = (long long *) allocationAddress;

		clear();
	}

	//! \brief Empty hardware counter structures
	inline void clear() override
	{
		const size_t numCounters = PAPIHardwareCounters::getNumEnabledCounters();

		memset(_countersDelta, 0, numCounters * sizeof(long long));
	}

	inline long long *getCountersBuffer()
	{
		return _countersDelta;
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
	inline uint64_t getAccumulated(HWCounters::counters_t) override
	{
		return 0;
	}

	//! \brief Retreive the size needed for hardware counters
	static inline size_t getTaskHardwareCountersSize()
	{
		const size_t numCounters = PAPIHardwareCounters::getNumEnabledCounters();

		return numCounters * sizeof(long long);
	}

};

#endif // PAPI_TASK_HARDWARE_COUNTERS_HPP
