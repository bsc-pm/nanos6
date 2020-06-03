/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_CPU_HARDWARE_COUNTERS_HPP
#define PQOS_CPU_HARDWARE_COUNTERS_HPP

#include <pqos.h>

#include "PQoSTaskHardwareCounters.hpp"
#include "hardware-counters/CPUHardwareCountersInterface.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


class PQoSCPUHardwareCounters : public CPUHardwareCountersInterface {

private:

	//! An array of regular HW counter deltas
	size_t _regularCounters[SupportPQoS::num_regular_counters];

	//! An array of accumulators of accumulating HW counters
	SupportPQoS::counter_accumulator_t _accumulatingCounters[SupportPQoS::num_accumulating_counters];

public:

	inline PQoSCPUHardwareCounters()
	{
		for (size_t id = 0; id < SupportPQoS::num_regular_counters; ++id) {
			_regularCounters[id] = 0;
		}
	}

	//! \brief Get the delta value of a hardware counter
	//!
	//! \param[in] counterId The type of counter to get the delta from
	inline double getDelta(HWCounters::counters_t counterId) override
	{
		switch (counterId) {
			case HWCounters::PQOS_MON_EVENT_L3_OCCUP:
				return (double) BoostAcc::mean(_accumulatingCounters[SupportPQoS::llc_usage]);
			case HWCounters::PQOS_PERF_EVENT_IPC:
				return (double) _regularCounters[SupportPQoS::ipc_retired] /
					(double) _regularCounters[SupportPQoS::ipc_unhalted];
			case HWCounters::PQOS_MON_EVENT_LMEM_BW:
				return (double) _regularCounters[SupportPQoS::mbm_local];
			case HWCounters::PQOS_MON_EVENT_RMEM_BW:
				return (double) _regularCounters[SupportPQoS::mbm_remote];
			case HWCounters::PQOS_PERF_EVENT_LLC_MISS:
				return (double) _regularCounters[SupportPQoS::llc_misses] /
					(double) _regularCounters[SupportPQoS::ipc_retired];
			default:
				FatalErrorHandler::fail("Event with id '", counterId, "' not supported (PQoS)");
				return 0.0;
		}
	}

	//! \brief Read delta counters for the current CPU
	//!
	//! \param[in] data The pqos data from which to gather counters
	inline void readCounters(const pqos_mon_data *data)
	{
		// For regular counters, the delta values in 'data' are reset from the
		// thread and we only care about accumulating them when we stop reading
		_regularCounters[SupportPQoS::mbm_local] = data->values.mbm_local_delta;
		_regularCounters[SupportPQoS::mbm_remote] = data->values.mbm_remote_delta;
		_regularCounters[SupportPQoS::llc_misses] = data->values.llc_misses_delta;
		_regularCounters[SupportPQoS::ipc_retired] = data->values.ipc_retired_delta;
		_regularCounters[SupportPQoS::ipc_unhalted] = data->values.ipc_unhalted_delta;

		// For accumulating counters, we must accumulate at start and stop
		_accumulatingCounters[SupportPQoS::llc_usage](data->values.llc);
	}

};

#endif // PQOS_CPU_HARDWARE_COUNTERS_HPP
