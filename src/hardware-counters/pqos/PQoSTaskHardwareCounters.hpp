/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_TASK_HARDWARE_COUNTERS_HPP
#define PQOS_TASK_HARDWARE_COUNTERS_HPP

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <pqos.h>

#include "hardware-counters/TaskHardwareCounters.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


typedef boost::accumulators::accumulator_set<size_t, boost::accumulators::stats<boost::accumulators::tag::mean> > counter_accumulator_t;


class PQoSTaskHardwareCounters : public TaskHardwareCounters {

private:

	enum accumulating_counter_t {
		llc_usage = 0,
		num_accumulating_counters
	};

	enum regular_counter_t {
		mbm_local = 0,
		mbm_remote,
		llc_misses,
		ipc_retired,
		ipc_unhalted,
		num_regular_counters
	};

	//! Whether reading of HW counters for the task is active
	bool _active;

	//! Arrays of regular HW counter deltas and accumulations
	size_t _regularCountersDelta[num_regular_counters];
	size_t _regularCountersAccumulated[num_regular_counters];

	//! An array of accumulators of accumulating HW counters
	counter_accumulator_t _accumulatingCounters[num_accumulating_counters];

public:

	inline PQoSTaskHardwareCounters() :
		_active(false)
	{
		for (size_t id = 0; id < num_regular_counters; ++id) {
			_regularCountersDelta[id] = 0;
			_regularCountersAccumulated[id] = 0;
		}
	}

	//! \brief Check whether HW counters are being read for the task
	inline bool isActive() const
	{
		return _active;
	}

	//! \brief Start reading HW counters for the current task
	//! \param[in] data The pqos data from which to gather counters
	inline void startReading(const pqos_mon_data *data)
	{
		assert(data != nullptr);

		_active = true;

		// For regular counters, the delta values in 'data' are reset from the
		// thread and we only care about accumulating them when we stop reading
		_regularCountersDelta[mbm_local] = data->values.mbm_local_delta;
		_regularCountersDelta[mbm_remote] = data->values.mbm_remote_delta;
		_regularCountersDelta[llc_misses] = data->values.llc_misses_delta;
		_regularCountersDelta[ipc_retired] = data->values.ipc_retired_delta;
		_regularCountersDelta[ipc_unhalted] = data->values.ipc_unhalted_delta;

		// For accumulating counters, we must accumulate at start and stop
		_accumulatingCounters[llc_usage](data->values.llc);
	}

	//! \brief Stop reading HW counters for the current task
	//! \param[in] data The pqos data from which to gather counters
	inline void stopReading(const pqos_mon_data *data)
	{
		assert(data != nullptr);

		_active = false;

		// For regular counters, the delta values in 'data' hold the counters
		// from start to stop, and those are the ones we want to read
		_regularCountersDelta[mbm_local] = data->values.mbm_local_delta;
		_regularCountersDelta[mbm_remote] = data->values.mbm_remote_delta;
		_regularCountersDelta[llc_misses] = data->values.llc_misses_delta;
		_regularCountersDelta[ipc_retired] = data->values.ipc_retired_delta;
		_regularCountersDelta[ipc_unhalted] = data->values.ipc_unhalted_delta;

		_regularCountersAccumulated[mbm_local] += _regularCountersDelta[mbm_local];
		_regularCountersAccumulated[mbm_remote] += _regularCountersDelta[mbm_remote];
		_regularCountersAccumulated[llc_misses] += _regularCountersDelta[llc_misses];
		_regularCountersAccumulated[ipc_retired] += _regularCountersDelta[ipc_retired];
		_regularCountersAccumulated[ipc_unhalted] += _regularCountersDelta[ipc_unhalted];

		// For accumulating counters, we must accumulate at start and stop
		_accumulatingCounters[llc_usage](data->values.llc);
	}

	//! \brief Get the delta value of a HW counter
	//! \param[in] counterType The type of counter to get the delta from
	inline double getDelta(HWCounters::counters_t counterType)
	{
		switch (counterType) {
			case HWCounters::llc_usage:
				return boost::accumulators::mean(_accumulatingCounters[llc_usage]);
			case HWCounters::ipc:
				return (double) _regularCountersDelta[ipc_retired] / (double) _regularCountersDelta[ipc_unhalted];
			case HWCounters::local_mem_bandwidth:
				return (double) _regularCountersDelta[mbm_local];
			case HWCounters::remote_mem_bandwidth:
				return (double) _regularCountersDelta[mbm_remote];
			case HWCounters::llc_miss_rate:
				return (double) _regularCountersDelta[llc_misses] / (double) _regularCountersDelta[ipc_retired];
			default:
				FatalErrorHandler::failIf(true, "Event with id '", counterType, "' not supported (PQoS)");
				return 0.0;
		}
	}

	//! \brief Get the accumulated value of a HW counter
	//! \param[in] counterType The type of counter to get the accumulation from
	inline double getAccumulated(HWCounters::counters_t counterType)
	{
		switch (counterType) {
			case HWCounters::llc_usage:
				return boost::accumulators::mean(_accumulatingCounters[llc_usage]);
			case HWCounters::ipc:
				return (double) _regularCountersAccumulated[ipc_retired] / (double) _regularCountersAccumulated[ipc_unhalted];
			case HWCounters::local_mem_bandwidth:
				return (double) _regularCountersAccumulated[mbm_local];
			case HWCounters::remote_mem_bandwidth:
				return (double) _regularCountersAccumulated[mbm_remote];
			case HWCounters::llc_miss_rate:
				return (double) _regularCountersAccumulated[llc_misses] / (double) _regularCountersAccumulated[ipc_retired];
			default:
				FatalErrorHandler::failIf(true, "Event with id '", counterType, "' not supported (PQoS)");
				return 0.0;
		}
	}

};

#endif // PQOS_TASK_HARDWARE_COUNTERS_HPP
