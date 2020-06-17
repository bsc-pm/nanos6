/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef HARDWARE_COUNTERS_HPP
#define HARDWARE_COUNTERS_HPP

#include <vector>

#include "HardwareCountersInterface.hpp"
#include "SupportedHardwareCounters.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


class Task;

class HardwareCounters {

private:

	//! Whether the verbose mode is enabled
	static EnvironmentVariable<bool> _verbose;

	//! The file where output must be saved when verbose mode is enabled
	static EnvironmentVariable<std::string> _verboseFile;

	//! The underlying PAPI backend
	static HardwareCountersInterface *_papiBackend;

	//! The underlying PQoS backend
	static HardwareCountersInterface *_pqosBackend;

	//! The underlying RAPL backend
	static HardwareCountersInterface *_raplBackend;

	//! Whether there is at least one enabled backend
	static bool _anyBackendEnabled;

	//! Whether each backend is enabled
	static std::vector<bool> _enabled;

	//! Enabled counters by the user
	static std::vector<HWCounters::counters_t> _enabledCounters;

private:

	//! \brief Load backend and counter enabling configuration from the default
	//! configuration file
	static void loadConfigurationFile();

	//! \brief Check if two or more backends are enabled and incompatible
	static inline void checkIncompatibleBackends()
	{
		if (_enabled[HWCounters::PAPI_BACKEND] && _enabled[HWCounters::PQOS_BACKEND]) {
			FatalErrorHandler::fail("PAPI and PQoS are incompatible hardware counter libraries");
		}
	}

public:

	//! \brief Pre-initialize the hardware counters API before hardware and CPUs are detected
	static void preinitialize();

	//! \brief Finish initializing the hardware counters API
	static void initialize();

	//! \brief Shutdown the hardware counters API
	static void shutdown();

	//! \brief Check whether a backend is enabled
	//!
	//! \param[in] backend The backend's id
	static inline bool isBackendEnabled(HWCounters::backends_t backend)
	{
		return _enabled[backend];
	}

	//! \brief Out of all the supported events, get the currently enabled ones
	static inline const std::vector<HWCounters::counters_t> &getEnabledCounters()
	{
		return _enabledCounters;
	}

	//! \brief Initialize hardware counter structures for a new thread
	static void threadInitialized();

	//! \brief Destroy the hardware counter structures of a thread
	static void threadShutdown();

	//! \brief Initialize hardware counter structures for a task
	//!
	//! \param[out] task The task to create structures for
	//! \param[in] enabled Whether to create structures and monitor this task
	static void taskCreated(Task *task, bool enabled = true);

	//! \brief Reinitialize all hardware counter structures for a task
	//!
	//! \param[out] task The task to reinitialize structures for
	static void taskReinitialized(Task *task);

	//! \brief Read and update hardware counters for a task
	//!
	//! \param[out] task The task to read hardware counters for
	static void updateTaskCounters(Task *task);

	//! \brief Read and update hardware counters for the runtime (current CPU)
	static void updateRuntimeCounters();

};

#endif // HARDWARE_COUNTERS_HPP
