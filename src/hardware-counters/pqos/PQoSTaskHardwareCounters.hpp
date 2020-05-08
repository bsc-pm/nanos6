/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_TASK_HARDWARE_COUNTERS_HPP
#define PQOS_TASK_HARDWARE_COUNTERS_HPP

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <pqos.h>

#include "hardware-counters/SupportedHardwareCounters.hpp"
#include "hardware-counters/TaskHardwareCountersInterface.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


// Public so PQoSCPUHardwareCounters can benefit from it
namespace BoostAcc = boost::accumulators;
namespace BoostAccTag = boost::accumulators::tag;

namespace SupportPQoS {
	typedef BoostAcc::accumulator_set<size_t, BoostAcc::stats<BoostAccTag::mean> > counter_accumulator_t;

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
}

class PQoSTaskHardwareCounters : public TaskHardwareCountersInterface {

private:

	//! Whether reading of HW counters for the task is active
	bool _active;

	//! Whether monitoring of counters for this task is enabled
	bool _enabled;

	//! Arrays of regular HW counter deltas and accumulations
	size_t _regularCountersDelta[SupportPQoS::num_regular_counters];
	size_t _regularCountersAccumulated[SupportPQoS::num_regular_counters];

	//! An array of accumulators of accumulating HW counters
	SupportPQoS::counter_accumulator_t _accumulatingCounters[SupportPQoS::num_accumulating_counters];

public:

	inline PQoSTaskHardwareCounters(bool enabled = true) :
		_active(false),
		_enabled(enabled)
	{
		for (size_t id = 0; id < SupportPQoS::num_regular_counters; ++id) {
			_regularCountersDelta[id] = 0;
			_regularCountersAccumulated[id] = 0;
		}
	}

	//! \brief Reset all structures to their default value
	inline void clear()
	{
		_active = false;

		for (size_t id = 0; id < SupportPQoS::num_regular_counters; ++id) {
			_regularCountersDelta[id] = 0;
			_regularCountersAccumulated[id] = 0;
		}

		for (size_t id = 0; id < SupportPQoS::num_accumulating_counters; ++id) {
			_accumulatingCounters[id] = {};
		}
	}

	//! \brief Enable or disable hardware counter monitoring for this task
	inline void setEnabled(bool enabled = true)
	{
		_enabled = enabled;
	}

	//! \brief Check whether hardware counter monitoring is enabled for this task
	inline bool isEnabled() const
	{
		return _enabled;
	}

	//! \brief Check whether hardware counters are being read for the task
	inline bool isActive() const
	{
		return _active;
	}

	//! \brief Start reading hardware counters for the current task
	//! \param[in] data The pqos data from which to gather counters
	inline void startReading(const pqos_mon_data *data)
	{
		assert(data != nullptr);

		_active = true;

		// For regular counters, the delta values in 'data' are reset from the
		// thread and we only care about accumulating them when we stop reading
		_regularCountersDelta[SupportPQoS::mbm_local] = data->values.mbm_local_delta;
		_regularCountersDelta[SupportPQoS::mbm_remote] = data->values.mbm_remote_delta;
		_regularCountersDelta[SupportPQoS::llc_misses] = data->values.llc_misses_delta;
		_regularCountersDelta[SupportPQoS::ipc_retired] = data->values.ipc_retired_delta;
		_regularCountersDelta[SupportPQoS::ipc_unhalted] = data->values.ipc_unhalted_delta;

		// For accumulating counters, we must accumulate at start and stop
		_accumulatingCounters[SupportPQoS::llc_usage](data->values.llc);
	}

	//! \brief Stop reading hardware counters for the current task
	//! \param[in] data The pqos data from which to gather counters
	inline void stopReading(const pqos_mon_data *data)
	{
		assert(data != nullptr);

		_active = false;

		// For regular counters, the delta values in 'data' hold the counters
		// from start to stop, and those are the ones we want to read
		_regularCountersDelta[SupportPQoS::mbm_local] = data->values.mbm_local_delta;
		_regularCountersDelta[SupportPQoS::mbm_remote] = data->values.mbm_remote_delta;
		_regularCountersDelta[SupportPQoS::llc_misses] = data->values.llc_misses_delta;
		_regularCountersDelta[SupportPQoS::ipc_retired] = data->values.ipc_retired_delta;
		_regularCountersDelta[SupportPQoS::ipc_unhalted] = data->values.ipc_unhalted_delta;

		_regularCountersAccumulated[SupportPQoS::mbm_local] += _regularCountersDelta[SupportPQoS::mbm_local];
		_regularCountersAccumulated[SupportPQoS::mbm_remote] += _regularCountersDelta[SupportPQoS::mbm_remote];
		_regularCountersAccumulated[SupportPQoS::llc_misses] += _regularCountersDelta[SupportPQoS::llc_misses];
		_regularCountersAccumulated[SupportPQoS::ipc_retired] += _regularCountersDelta[SupportPQoS::ipc_retired];
		_regularCountersAccumulated[SupportPQoS::ipc_unhalted] += _regularCountersDelta[SupportPQoS::ipc_unhalted];

		// For accumulating counters, we must accumulate at start and stop
		_accumulatingCounters[SupportPQoS::llc_usage](data->values.llc);
	}

	//! \brief Get the delta value of a hardware counter
	//! \param[in] counterId The type of counter to get the delta from
	inline double getDelta(HWCounters::counters_t counterId)
	{
		switch (counterId) {
			case HWCounters::PQOS_MON_EVENT_L3_OCCUP:
				return (double) BoostAcc::mean(_accumulatingCounters[SupportPQoS::llc_usage]);
			case HWCounters::PQOS_PERF_EVENT_IPC:
				return (double) _regularCountersDelta[SupportPQoS::ipc_retired] /
					(double) _regularCountersDelta[SupportPQoS::ipc_unhalted];
			case HWCounters::PQOS_MON_EVENT_LMEM_BW:
				return (double) _regularCountersDelta[SupportPQoS::mbm_local];
			case HWCounters::PQOS_MON_EVENT_RMEM_BW:
				return (double) _regularCountersDelta[SupportPQoS::mbm_remote];
			case HWCounters::PQOS_PERF_EVENT_LLC_MISS:
				return (double) _regularCountersDelta[SupportPQoS::llc_misses] /
					(double) _regularCountersDelta[SupportPQoS::ipc_retired];
			default:
				FatalErrorHandler::fail("Event with id '", counterId, "' not supported (PQoS)");
				return 0.0;
		}
	}

	//! \brief Get the accumulated value of a hardware counter
	//! \param[in] counterId The type of counter to get the accumulation from
	inline double getAccumulated(HWCounters::counters_t counterId)
	{
		switch (counterId) {
			case HWCounters::PQOS_MON_EVENT_L3_OCCUP:
				return (double) BoostAcc::mean(_accumulatingCounters[SupportPQoS::llc_usage]);
			case HWCounters::PQOS_PERF_EVENT_IPC:
				return (double) _regularCountersAccumulated[SupportPQoS::ipc_retired] /
					(double) _regularCountersAccumulated[SupportPQoS::ipc_unhalted];
			case HWCounters::PQOS_MON_EVENT_LMEM_BW:
				return (double) _regularCountersAccumulated[SupportPQoS::mbm_local];
			case HWCounters::PQOS_MON_EVENT_RMEM_BW:
				return (double) _regularCountersAccumulated[SupportPQoS::mbm_remote];
			case HWCounters::PQOS_PERF_EVENT_LLC_MISS:
				return (double) _regularCountersAccumulated[SupportPQoS::llc_misses] /
					(double) _regularCountersAccumulated[SupportPQoS::ipc_retired];
			default:
				FatalErrorHandler::fail("Event with id '", counterId, "' not supported (PQoS)");
				return 0.0;
		}
	}

};

#endif // PQOS_TASK_HARDWARE_COUNTERS_HPP
