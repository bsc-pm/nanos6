/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef HARDWARE_COUNTERS_HPP
#define HARDWARE_COUNTERS_HPP

#include "HardwareCountersInterface.hpp"
#include "SupportedHardwareCounters.hpp"
#if HAVE_PAPI
	#include "hardware-counters/papi/PAPIHardwareCountersImplementation.hpp"
#endif
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

	//! \brief Initialize the hardware counters API with the correct backend
	static inline void initialize()
	{
		assert(_hwCountersInterface == nullptr);

		if (_chosenBackend.getValue() == "papi") {
#if HAVE_PAPI
			_hwCountersInterface = new PAPIHardwareCountersImplementation();
#else
			_hwCountersInterface = new NullHardwareCountersImplementation();
			FatalErrorHandler::warnIf(true, "PAPI library not found, disabling hardware counters.");
#endif
		} else if (_chosenBackend.getValue() == "pqos") {
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

	//! \brief Shutdown the hardware counters API
	static inline void shutdown()
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->shutdown();

		delete _hwCountersInterface;
	}

	//! \brief Retrieve the chosen hardware counters backend
	//! \return A string with the chosen backend
	static inline std::string getChosenBackend()
	{
		return _chosenBackend.getValue();
	}

	//! \brief Check whether a type of counter is supported
	//! \param[in] counterType The type of hardware counter
	static inline bool isSupported(HWCounters::counters_t counterType)
	{
		assert(_hwCountersInterface != nullptr);

		return _hwCountersInterface->isSupported(counterType);
	}

	//! \brief Initialize hardware counter structures for a new thread
	static inline void threadInitialized()
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->threadInitialized();
	}

	//! \brief Destroy the hardware counter structures of a thread
	static inline void threadShutdown()
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->threadShutdown();
	}

	//! \brief Initialize hardware counter structures for a task
	//! \param[in,out] task The task to create structures for
	//! \param[in] enabled Whether to create structures and monitor this task
	static inline void taskCreated(Task *task, bool enabled = true)
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->taskCreated(task, enabled);
	}

	//! \brief Reinitialize all hardware counter structures for a task
	//! \param[in,out] task The task to reinitialize structures for
	static inline void taskReinitialized(Task *task)
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->taskReinitialized(task);
	}

	//! \brief Start reading hardware counters for a task
	//! \param[in,out] task The task to start hardware counter monitoring for
	static inline void taskStarted(Task *task)
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->taskStarted(task);
	}

	//! \brief Stop reading hardware counters for a task
	//! \param[in,out] task The task to stop hardware counters monitoring for
	static inline void taskStopped(Task *task)
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->taskStopped(task);
	}

	//! \brief Finish monitoring a task's hardware counters and accumulate them
	//! \param[in,out] task The task to finish hardware counters monitoring for
	static inline void taskFinished(Task *task)
	{
		assert(_hwCountersInterface != nullptr);

		_hwCountersInterface->taskFinished(task);
	}

	//! \brief Get the size of task hardware counter structures for the chosen
	//! backend
	static inline size_t getTaskHardwareCountersSize()
	{
		assert(_hwCountersInterface != nullptr);

		return _hwCountersInterface->getTaskHardwareCountersSize();
	}

};

#endif // HARDWARE_COUNTERS_HPP
