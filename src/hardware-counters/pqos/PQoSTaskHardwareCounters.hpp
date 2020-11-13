/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_TASK_HARDWARE_COUNTERS_HPP
#define PQOS_TASK_HARDWARE_COUNTERS_HPP

#include <pqos.h>
#include <vector>

#include "PQoSHardwareCounters.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"
#include "hardware-counters/TaskHardwareCountersInterface.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


class PQoSTaskHardwareCounters : public TaskHardwareCountersInterface {

private:

	//! Arrays of hardware counter deltas and accumulated values
	uint64_t *_counterDelta;
	uint64_t *_counterAccumulated;

	//! Number of samples used to average L3 occupancy
	size_t _numSamples;

public:

	inline PQoSTaskHardwareCounters(void *allocationAddress) :
		_numSamples(0)
	{
		assert(allocationAddress != nullptr);

		const size_t numEvents = PQoSHardwareCounters::getNumEnabledCounters();
		_counterDelta = (uint64_t *) allocationAddress;
		_counterAccumulated = (uint64_t *) ((char *) allocationAddress + (numEvents * sizeof(uint64_t)));

		for (size_t id = 0; id < numEvents; ++id) {
			_counterDelta[id] = 0;
			_counterAccumulated[id] = 0;
		}
	}

	//! \brief Reset all structures to their default value
	inline void clear() override
	{
		assert(_counterDelta != nullptr);
		assert(_counterAccumulated != nullptr);

		const size_t numEvents = PQoSHardwareCounters::getNumEnabledCounters();
		for (size_t id = 0; id < numEvents; ++id) {
			_counterDelta[id] = 0;
			_counterAccumulated[id] = 0;
		}
	}

	//! \brief Read hardware counters for the current task
	//!
	//! \param[in] data The pqos data from which to gather counters
	inline void readCounters(const pqos_mon_data *data)
	{
		assert(data != nullptr);

		for (size_t id = HWCounters::HWC_PQOS_MIN_EVENT; id <= HWCounters::HWC_PQOS_MAX_EVENT; ++id) {
			if (PQoSHardwareCounters::isCounterEnabled((HWCounters::counters_t) id)) {
				int innerId = PQoSHardwareCounters::getInnerIdentifier((HWCounters::counters_t) id);
				assert(innerId >= 0);

				switch (id) {
					case HWCounters::HWC_PQOS_MON_EVENT_L3_OCCUP:
						++_numSamples;
						_counterDelta[innerId] =
							((double) _counterDelta[innerId] * (_numSamples - 1) + data->values.llc) / _numSamples;

						_counterAccumulated[innerId] = _counterDelta[innerId];
						break;
					case HWCounters::HWC_PQOS_MON_EVENT_LMEM_BW:
						_counterDelta[innerId] = data->values.mbm_local_delta;
						_counterAccumulated[innerId] += _counterDelta[innerId];
						break;
					case HWCounters::HWC_PQOS_MON_EVENT_RMEM_BW:
						_counterDelta[innerId] = data->values.mbm_remote_delta;
						_counterAccumulated[innerId] += _counterDelta[innerId];
						break;
					case HWCounters::HWC_PQOS_PERF_EVENT_LLC_MISS:
						_counterDelta[innerId] = data->values.llc_misses_delta;
						_counterAccumulated[innerId] += _counterDelta[innerId];
						break;
					case HWCounters::HWC_PQOS_PERF_EVENT_INSTRUCTIONS:
						_counterDelta[innerId] = data->values.ipc_retired_delta;
						_counterAccumulated[innerId] += _counterDelta[innerId];
						break;
					case HWCounters::HWC_PQOS_PERF_EVENT_CYCLES:
						_counterDelta[innerId] = data->values.ipc_unhalted_delta;
						_counterAccumulated[innerId] += _counterDelta[innerId];
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
	inline uint64_t getDelta(HWCounters::counters_t counterType) const override
	{
		assert(PQoSHardwareCounters::isCounterEnabled(counterType));

		int innerId = PQoSHardwareCounters::getInnerIdentifier(counterType);
		assert(innerId >= 0 && (size_t) innerId < PQoSHardwareCounters::getNumEnabledCounters());

		return _counterDelta[innerId];
	}

	//! \brief Get the accumulated value of a hardware counter
	//!
	//! \param[in] counterType The type of counter to get the accumulation from
	inline uint64_t getAccumulated(HWCounters::counters_t counterType) const override
	{
		assert(PQoSHardwareCounters::isCounterEnabled(counterType));

		int innerId = PQoSHardwareCounters::getInnerIdentifier(counterType);
		assert(innerId >= 0 && (size_t) innerId < PQoSHardwareCounters::getNumEnabledCounters());

		return _counterAccumulated[innerId];
	}

	//! \brief Combine the counters of two tasks
	//!
	//! \param[in] combineeCounters The counters of a task, which will be combined into
	//! the current counters
	inline void combineCounters(const TaskHardwareCountersInterface *combineeCounters) override
	{
		PQoSTaskHardwareCounters *childCounters = (PQoSTaskHardwareCounters *) combineeCounters;
		assert(childCounters != nullptr);
		assert(_counterDelta != nullptr);
		assert(_counterAccumulated != nullptr);

		// Get the raw data of each regular counter and combine it
		for (size_t id = HWCounters::HWC_PQOS_MIN_EVENT; id <= HWCounters::HWC_PQOS_MAX_EVENT; ++id) {
			HWCounters::counters_t counterType = (HWCounters::counters_t) id;
			if (PQoSHardwareCounters::isCounterEnabled(counterType)) {
				int innerId = PQoSHardwareCounters::getInnerIdentifier(counterType);
				assert(innerId >= 0);

				switch (counterType) {
					case HWCounters::HWC_PQOS_MON_EVENT_L3_OCCUP:
						// Only take it into account if the child (most likely taskfor
						// collaborator) truly participated in the execution
						uint64_t childValue = childCounters->getDelta(counterType);
						if (childValue != 0) {
							++_numSamples;
							_counterDelta[innerId] =
								((double) _counterDelta[innerId] * (_numSamples - 1) + childValue) / _numSamples;

							_counterAccumulated[innerId] = _counterDelta[innerId];
						}
						break;
					default:
						_counterDelta[innerId] = childCounters->getDelta(counterType);
						_counterAccumulated[innerId] += childCounters->getAccumulated(counterType);
						break;
				}
			}
		}
	}

	//! \brief Retreive the size needed for hardware counters
	static inline size_t getTaskHardwareCountersSize()
	{
		// We need 2 times the number of events for the delta and accumulated values
		const size_t numEvents = PQoSHardwareCounters::getNumEnabledCounters();
		return (numEvents * 2 * sizeof(uint64_t));
	}

};

#endif // PQOS_TASK_HARDWARE_COUNTERS_HPP
