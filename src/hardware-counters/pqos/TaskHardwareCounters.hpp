/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_TASK_HARDWARE_COUNTERS_HPP
#define PQOS_TASK_HARDWARE_COUNTERS_HPP

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <pqos.h>

#include "hardware-counters/SupportedHardwareCounters.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

typedef boost::accumulators::accumulator_set<size_t, boost::accumulators::stats<boost::accumulators::tag::mean> > counter_accumulator_t;

#define DEFAULT_COST 1


struct SnapshotCounter {
	
	//! Temporary value for when a task is stopped/paused
	size_t _paused;
	
	//! Temporary value for when a task is started/resumed
	size_t _resumed;
	
	//! The accumulated value for snapshot counters
	counter_accumulator_t _accumulated;
	
	
	inline SnapshotCounter() :
		_paused(0),
		_resumed(0),
		_accumulated()
	{
	}
};

struct AccumulatingCounter {
	
	//! Temporary value for when a task is stopped/paused
	size_t _paused;
	
	//! Temporary value for when a task is started/resumed
	size_t _resumed;
	
	//! The accumulated value for accumulating counters
	size_t _accumulated;
	
	
	inline AccumulatingCounter() :
		_paused(0),
		_resumed(0),
		_accumulated(0)
	{
	}
};


class TaskHardwareCounters {

private:
	
	enum snapshot_counters_t {
		llc_usage = 0,
		num_snapshot_counters
	};
	
	enum accumulating_counters_t {
		mbm_local = 0,
		mbm_remote,
		llc_misses,
		ipc_retired,
		ipc_unhalted,
		num_accumulating_counters
	};
	
	//! Identifies the tasktype
	std::string _label;
	
	//! The computational cost of the task
	size_t _cost;
	
	//! Whether the task is currently monitoring hardware counters
	bool _currentlyMonitoring;
	
	//! Task-specific PQoS counters
	SnapshotCounter _snapshotCounters[num_snapshot_counters];
	AccumulatingCounter _accumulatingCounters[num_accumulating_counters];
	
	
public:
	
	inline TaskHardwareCounters() :
		_cost(DEFAULT_COST),
		_currentlyMonitoring(false)
	{
	}
	
	
	//    SETTERS & GETTERS    //
	
	inline void setLabel(const std::string &label)
	{
		_label = label;
	}
	
	inline const std::string &getLabel() const
	{
		return _label;
	}
	
	inline void setCost(size_t cost)
	{
		if (cost > 0) {
			_cost = cost;
		}
	}
	
	inline size_t getCost() const
	{
		return _cost;
	}
	
	inline bool isCurrentlyMonitoring() const
	{
		return _currentlyMonitoring;
	}
	
	
	//    COUNTER MANIPULATION    //
	
	//! \brief Gather counters when a task is resumed or started
	//! \param data The pqos data from which to gather counters
	inline void startOrResume(const pqos_mon_data *data)
	{
		assert(data != nullptr);
		
		_currentlyMonitoring = true;
		
		_snapshotCounters[llc_usage]._resumed        = data->values.llc;
		_accumulatingCounters[mbm_local]._resumed    = data->values.mbm_local_delta;
		_accumulatingCounters[mbm_remote]._resumed   = data->values.mbm_remote_delta;
		_accumulatingCounters[llc_misses]._resumed   = data->values.llc_misses_delta;
		_accumulatingCounters[ipc_retired]._resumed  = data->values.ipc_retired_delta;
		_accumulatingCounters[ipc_unhalted]._resumed = data->values.ipc_unhalted_delta;
	}
	
	//! \brief Gather counters when a task is stopped or paused
	//! \param data The pqos data from which to gather counters
	inline void stopOrPause(const pqos_mon_data *data)
	{
		assert(data != nullptr);
		
		_currentlyMonitoring = false;
		
		_snapshotCounters[llc_usage]._paused        = data->values.llc;
		_accumulatingCounters[mbm_local]._paused    = data->values.mbm_local_delta;
		_accumulatingCounters[mbm_remote]._paused   = data->values.mbm_remote_delta;
		_accumulatingCounters[llc_misses]._paused   = data->values.llc_misses_delta;
		_accumulatingCounters[ipc_retired]._paused  = data->values.ipc_retired_delta;
		_accumulatingCounters[ipc_unhalted]._paused = data->values.ipc_unhalted_delta;
	}
	
	//! \brief Accumulate counters
	inline void accumulateCounters()
	{
		// For snapshot-action counters, when tasks start, resume, stop, or
		// pause, the value obtained indicates the counter's value at that
		// exact moment. To obtain a single value, we sample every start, stop
		// pause and resume point and then average all of the obtained values
		// through accumulators.
		// This is needed (i.e. in LLC Usage) since having a value of "X" at
		// two different moments in time may mean that the LLC usage has not
		// changed, or that it has changed and it is back at value "X"
		for (unsigned short i = 0; i < num_snapshot_counters; ++i) {
			_snapshotCounters[i]._accumulated(_snapshotCounters[i]._paused);
			_snapshotCounters[i]._accumulated(_snapshotCounters[i]._resumed);
		}
		
		// For accumulated events, when tasks start, resume, stop or pause
		// execution, the value obtained indicates the delta value from this
		// exact moment w.r.t. the previous polled valued. These "delta" values
		// are aggregated to obtain the total value of the event
		for (unsigned short i = 0; i < num_accumulating_counters; ++i) {
			_accumulatingCounters[i]._accumulated += _accumulatingCounters[i]._paused;
		}
	}
	
	//! \brief Get the value of a hardware counter
	//! \param counterId The id of the counter
	inline double getCounter(HWCounters::counters_t counterId)
	{
		switch (counterId) {
			case HWCounters::llc_usage :
				return boost::accumulators::mean(_snapshotCounters[llc_usage]._accumulated);
			case HWCounters::ipc :
				return (double) _accumulatingCounters[ipc_retired]._accumulated / (double) _accumulatingCounters[ipc_unhalted]._accumulated;
			case HWCounters::local_mem_bandwidth :
				return (double) _accumulatingCounters[mbm_local]._accumulated; // KB
			case HWCounters::remote_mem_bandwidth :
				return (double) _accumulatingCounters[mbm_remote]._accumulated; // KB
			case HWCounters::llc_miss_rate :
				return (double) _accumulatingCounters[llc_misses]._accumulated / (double) _accumulatingCounters[ipc_retired]._accumulated;
			default :
				FatalErrorHandler::failIf(true, "PQoS does not support Hardware Counter with id: ", counterId);
				return HWCounters::invalid_counter;
		}
		return HWCounters::invalid_counter;
	}
	
};

#endif // PQOS_TASK_HARDWARE_COUNTERS_HPP
