/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_CPU_HARDWARE_COUNTERS_HPP
#define PQOS_CPU_HARDWARE_COUNTERS_HPP

#include <pqos.h>

#include "PQoSHardwareCounters.hpp"
#include "hardware-counters/CPUHardwareCountersInterface.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"


class PQoSCPUHardwareCounters : public CPUHardwareCountersInterface {

private:

	//! The array of hardware counter deltas
	uint64_t _counters[HWCounters::PQOS_MAX_EVENT - HWCounters::PQOS_MIN_EVENT];

	//! Number of samples taken into account for the average of L3 occupancy
	size_t _numSamples;

public:

	inline PQoSCPUHardwareCounters() :
		_numSamples(0)
	{
		for (size_t id = 0; id < HWCounters::PQOS_MAX_EVENT - HWCounters::PQOS_MIN_EVENT; ++id) {
			_counters[id] = 0;
		}
	}

	//! \brief Read delta counters for the current CPU
	//!
	//! \param[in] data The pqos data from which to gather counters
	inline void readCounters(const pqos_mon_data *data)
	{
		// For regular counters, the delta values in 'data' are reset when
		// needed, so we simply copy the new deltas. In the case of L3
		// occupancy, we compute an average in-place
		for (size_t id = HWCounters::PQOS_MIN_EVENT; id < HWCounters::PQOS_MAX_EVENT; ++id) {
			// Make sure the event is enabled
			if (PQoSHardwareCounters::isCounterEnabled((HWCounters::counters_t) id)) {
				size_t innerId = id - HWCounters::PQOS_MIN_EVENT;
				switch (id) {
					case HWCounters::PQOS_MON_EVENT_L3_OCCUP:
						if (!_numSamples) {
							++_numSamples;
							_counters[innerId] = data->values.llc;
						} else {
							++_numSamples;
							_counters[innerId] =
								((double) _counters[innerId] * (_numSamples - 1) + data->values.llc) / _numSamples;
						}
						break;
					case HWCounters::PQOS_MON_EVENT_LMEM_BW:
						_counters[innerId] = data->values.mbm_local_delta;
						break;
					case HWCounters::PQOS_MON_EVENT_RMEM_BW:
						_counters[innerId] = data->values.mbm_remote_delta;
						break;
					case HWCounters::PQOS_PERF_EVENT_LLC_MISS:
						_counters[innerId] = data->values.llc_misses_delta;
						break;
					case HWCounters::PQOS_PERF_EVENT_RETIRED_INSTRUCTIONS:
						_counters[innerId] = data->values.ipc_retired_delta;
						break;
					case HWCounters::PQOS_PERF_EVENT_UNHALTED_CYCLES:
						_counters[innerId] = data->values.ipc_unhalted_delta;
						break;
					default:
						assert(false);
				}
			}
		}
	}

	//! \brief Get the delta value of a hardware counter
	//!
	//! \param[in] counterType The type of counter to get the delta from
	inline uint64_t getDelta(HWCounters::counters_t counterType) override
	{
		assert(counterType >= HWCounters::PQOS_MIN_EVENT);
		assert(counterType <= HWCounters::PQOS_MAX_EVENT);
		assert(PQoSHardwareCounters::isCounterEnabled(counterType));

		return _counters[counterType - HWCounters::PQOS_MIN_EVENT];
	}

};

#endif // PQOS_CPU_HARDWARE_COUNTERS_HPP
