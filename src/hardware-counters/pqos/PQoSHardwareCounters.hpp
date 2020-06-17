/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PQOS_HARDWARE_COUNTERS_HPP
#define PQOS_HARDWARE_COUNTERS_HPP

#include <pqos.h>
#include <vector>

#include "hardware-counters/CPUHardwareCountersInterface.hpp"
#include "hardware-counters/HardwareCountersInterface.hpp"
#include "hardware-counters/TaskHardwareCountersInterface.hpp"
#include "hardware-counters/ThreadHardwareCountersInterface.hpp"


#define DISABLED_PQOS_COUNTER -1

class PQoSHardwareCounters : public HardwareCountersInterface {

private:

	//! Whether PQoS HW Counter instrumentation is enabled
	bool _enabled;

	//! An enumeration containing the events that we monitor
	enum pqos_mon_event _monitoredEvents;

	//! The number of enabled counters (enabled by the user and available)
	static size_t _numEnabledCounters;

	//! Maps HWCounters::counters_t identifiers with the "inner pqos id" (0..N)
	//!
	//! NOTE: This is a vector with as many positions as possible counters in
	//! the pqos backend (PQOS_NUM_EVENTS), even those that are disabled.
	//!   - If the value of a position is -1, there is no mapping (i.e., the
	//!     event is disabled).
	//!   - On the other hand, the value of a position is a mapping of a general
	//!     ID (PQOS_MON_EVENT_LMEM_BW == 101) to the real position it should
	//!     occupy in a vector with only enabled pqos events (0 for instance if
	//!     this is the only PQoS enabled event)
	//! _idMap.size = PQOS_NUM_EVENTS
	//! _idMap[PQOS_MON_EVENT_LMEM_BW(101) - PQOS_MIN_EVENT(100)] = 0
	static std::vector<int> _idMap;

public:

	//! \brief Initializ the PQoS backend
	//!
	//! \param[in] verbose Whether verbose mode is enabled
	//! \param[in] verboseFile The file onto which to write verbose messages
	//! \param[in,out] enabledEvents A vector with all the events enabled by the user,
	//! which will be modified to disable those that are unavailable
	PQoSHardwareCounters(
		bool,
		const std::string &,
		std::vector<HWCounters::counters_t> &enabledEvents
	);

	~PQoSHardwareCounters();

	//! \brief Retreive the mapping from a counters_t identifier to the inner
	//! identifier of arrays with only enabled events
	//!
	//! \param[in] counterType The identifier to translate
	//! \return An integer with the relative position in arrays of only enabled
	//! events or DISABLED_PQOS_COUNTER (-1) if this counter is disabled
	static inline int getInnerIdentifier(HWCounters::counters_t counterType)
	{
		assert(_idMap.size() > 0);
		assert((counterType - HWCounters::PQOS_MIN_EVENT) < (int) _idMap.size());

		return _idMap[counterType - HWCounters::PQOS_MIN_EVENT];
	}

	//! \brief Check whether a counter is enabled
	//!
	//! \param[in] counterType The identifier to translate
	static inline bool isCounterEnabled(HWCounters::counters_t counterType)
	{
		assert(_idMap.size() > 0);
		assert((counterType - HWCounters::PQOS_MIN_EVENT) < (int) _idMap.size());

		return (_idMap[counterType - HWCounters::PQOS_MIN_EVENT] != DISABLED_PQOS_COUNTER);
	}

	//! \brief Get the number of enabled counters in the PQoS backend
	static inline size_t getNumEnabledCounters()
	{
		return _numEnabledCounters;
	}

	void threadInitialized(ThreadHardwareCountersInterface *threadCounters) override;

	void threadShutdown(ThreadHardwareCountersInterface *threadCounters) override;

	void taskReinitialized(TaskHardwareCountersInterface *taskCounters) override;

	void updateTaskCounters(
		ThreadHardwareCountersInterface *threadCounters,
		TaskHardwareCountersInterface *taskCounters
	) override;

	void updateRuntimeCounters(
		CPUHardwareCountersInterface *cpuCounters,
		ThreadHardwareCountersInterface *threadCounters
	) override;

};

#endif // PQOS_HARDWARE_COUNTERS_HPP
