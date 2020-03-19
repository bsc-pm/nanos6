/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef HARDWARE_COUNTERS_HPP
#define HARDWARE_COUNTERS_HPP

#include "HardwareCountersInterface.hpp"
#include "SupportedHardwareCounters.hpp"
//TODO: Uncomment when ready
//#include "hardware-counters/papi/PAPIHardwareCountersImplementation.hpp"
#if HAVE_PQOS
	#include "hardware-counters/pqos/PQoSHardwareCountersImplementation.hpp"
#endif
#include "hardware-counters/null/NullHardwareCountersImplementation.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


class Task;

class HardwareCounters {

private:

	//! An env var that shows the chosen backend (defaults to null)
	static EnvironmentVariable<std::string> _chosenBackend;

	//! The underlying implementation of the hardware counters backend
	static HardwareCountersInterface *_hwCountersInterface;

	//! Whether the verbose mode is enabled
	static EnvironmentVariable<bool> _verbose;

	//! The file where output must be saved when verbose mode is enabled
	static EnvironmentVariable<std::string> _verboseFile;


public:

	static inline void initialize()
	{
		assert(_hwCountersInterface == nullptr);

/* TODO: Uncomment when ready
		if (_chosenBackend.getValue() == "papi") {
#if HAVE_PAPI
			_hwCountersInterface = new PAPIHardwareCountersImplementation();
#else
			_hwCountersInterface = new NullHardwareCountersImplementation();
			FatalErrorHandler::warnIf(true, "PAPI library not found, disabling hardware counters.");
#endif
		} else if (_chosenBackend.getValue() == "pqos") {
*/
		if (_chosenBackend.getValue() == "pqos") {
#if HAVE_PQOS
			_hwCountersInterface = new PQoSHardwareCountersImplementation();
#else
			_hwCountersInterface = new NullHardwareCountersImplementation();
			FatalErrorHandler::warnIf(true, "PQoS library not found, disabling hardware counters.");
#endif
		} else if (_chosenBackend.getValue() == "null") {
			_hwCountersInterface = new NullHardwareCountersImplementation();
		} else {
			FatalErrorHandler::failIf(
				true,
				"Unexistent backend for hardware counters instrumentation: ",
				_chosenBackend.getValue()
			);
		}

		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->initialize(_verbose.getValue(), _verboseFile.getValue());
	}

	static inline void shutdown()
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->shutdown();

		delete _hwCountersInterface;
	}

	static inline std::string getChosenBackend()
	{
		return _chosenBackend.getValue();
	}

	static inline bool isSupported(HWCounters::counters_t counterType)
	{
		assert(_hwCountersInterface != nullptr);

		return _hwCountersInterface->isSupported(counterType);
	}

	static inline void threadInitialized()
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->threadInitialized();
	}

	static inline void threadShutdown()
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->threadShutdown();
	}

	static inline void taskCreated(Task *task, bool enabled = true)
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->taskCreated(task, enabled);
	}

	static inline void taskStarted(Task *task)
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->taskStarted(task);
	}

	static inline void taskStopped(Task *task)
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->taskStopped(task);
	}

	static inline void taskFinished(Task *task)
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->taskFinished(task);
	}

};

#endif // HARDWARE_COUNTERS_HPP
