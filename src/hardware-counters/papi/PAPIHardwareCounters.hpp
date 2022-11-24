/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef PAPI_HARDWARE_COUNTERS_HPP
#define PAPI_HARDWARE_COUNTERS_HPP

#include <cassert>
#include <string>
#include <vector>

#include "hardware-counters/HardwareCountersInterface.hpp"
#include "hardware-counters/SupportedHardwareCounters.hpp"


class CPUHardwareCountersInterface;
class TaskHardwareCountersInterface;
class ThreadHardwareCountersInterface;

class PAPIHardwareCounters : public HardwareCountersInterface {

private:

	//! Whether the PAPI HW Counter backend is enabled
	bool _enabled;

	//! Whether the verbose mode is enabled
	bool _verbose;

	//! A vector containing all the enabled PAPI event codes
	std::vector<int> _enabledPAPIEventCodes;

	//! The number of enabled counters (enabled by the user and available)
	static size_t _numEnabledCounters;

	//! Maps HWCounters::counters_t identifiers with the "inner PAPI id" (0..N)
	//!
	//! NOTE: This is an array with as many positions as possible counters in
	//! the PAPI backend (PAPI_NUM_EVENTS), even those that are disabled.
	//!   - If the value of a position is -1, there is no mapping (i.e., the
	//!     event is disabled).
	//!   - On the other hand, the value of a position is a mapping of a general
	//!     ID (HWC_PAPI_L1_DCM == 201) to the real position it should
	//!     occupy in a vector with only enabled PAPI events (0 for instance if
	//!     this is the only PAPI enabled event)
	//! _idMap[HWC_PAPI_L1_DCM(201) - HWC_PAPI_MIN_EVENT(200)] = 0
	static int _idMap[HWCounters::HWC_PAPI_NUM_EVENTS];

	static const int DISABLED_PAPI_COUNTER = -1;

private:

	void testMaximumNumberOfEvents();

public:

	//! \brief Initialize the PAPI backend
	//!
	//! \param[in] verbose Whether verbose mode is enabled
	//! \param[in] verboseFile The file onto which to write verbose messages
	//! \param[in,out] enabledEvents A vector with all the events enabled by the user,
	//! which will be modified to disable those that are unavailable
	PAPIHardwareCounters(
		bool verbose,
		const std::string &,
		std::vector<HWCounters::counters_t> &enabledEvents
	);

	inline ~PAPIHardwareCounters()
	{
	}

	//! \brief Retreive the mapping from a counters_t identifier to the inner
	//! identifier of arrays with only enabled events
	//!
	//! \param[in] counterType The identifier to translate
	//! \return An integer with the relative position in arrays of only enabled
	//! events or DISABLED_PAPI_COUNTER (-1) if this counter is disabled
	static inline int getInnerIdentifier(HWCounters::counters_t counterType)
	{
		assert((counterType - HWCounters::HWC_PAPI_MIN_EVENT) < HWCounters::HWC_PAPI_NUM_EVENTS);

		return _idMap[counterType - HWCounters::HWC_PAPI_MIN_EVENT];
	}

	//! \brief Check whether a counter is enabled
	//!
	//! \param[in] counterType The identifier to translate
	static inline bool isCounterEnabled(HWCounters::counters_t counterType)
	{
		assert((counterType - HWCounters::HWC_PAPI_MIN_EVENT) < HWCounters::HWC_PAPI_NUM_EVENTS);

		return (_idMap[counterType - HWCounters::HWC_PAPI_MIN_EVENT] != DISABLED_PAPI_COUNTER);
	}

	//! \brief Get the number of enabled counters in the PAPI backend
	static inline size_t getNumEnabledCounters()
	{
		return _numEnabledCounters;
	}

	void threadInitialized(ThreadHardwareCountersInterface *threadCounters) override;

	void threadShutdown(ThreadHardwareCountersInterface *) override;

	void updateTaskCounters(
		ThreadHardwareCountersInterface *threadCounters,
		TaskHardwareCountersInterface *taskCounters
	) override;

	void updateRuntimeCounters(
		CPUHardwareCountersInterface *cpuCounters,
		ThreadHardwareCountersInterface *threadCounters
	) override;

};

#endif // PAPI_HARDWARE_COUNTERS_HPP
