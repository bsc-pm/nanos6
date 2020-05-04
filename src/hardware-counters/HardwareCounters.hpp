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
#include "support/JsonFile.hpp"


class Task;

class HardwareCounters {

private:

	//! Whether the verbose mode is enabled
	static EnvironmentVariable<bool> _verbose;

	//! The file where output must be saved when verbose mode is enabled
	static EnvironmentVariable<std::string> _verboseFile;

	//! The underlying PQoS backend
	static HardwareCountersInterface *_pqosBackend;

	//! The underlying PAPI backend
	static HardwareCountersInterface *_papiBackend;

	//! Whether each backend is enabled
	static std::vector<bool> _enabled;

	//! Enabled events by the user (id, description)
	static std::vector<bool> _enabledEvents;

private:

	static inline void loadConfigurationFile()
	{
		JsonFile configFile = JsonFile("./nanos6_hwcounters.json");
		if (configFile.fileExists()) {
			configFile.loadData();

			// Navigate through the file and extract the enabled backens and counters
			configFile.getRootNode()->traverseChildrenNodes(
				[&](const std::string &category, const JsonNode<> &categoryNode) {
					if (category == "backends") {
						if (categoryNode.dataExists("papi")) {
							bool converted = false;
							bool enabled = categoryNode.getData("papi", converted);
							assert(converted);

							_enabled[HWCounters::PAPI_BACKEND] = enabled;
						}

						if (categoryNode.dataExists("pqos")) {
							bool converted = false;
							bool enabled = categoryNode.getData("pqos", converted);
							assert(converted);

							_enabled[HWCounters::PQOS_BACKEND] = enabled;
						}
					} else if (category == "counters") {
						// Check which events are enabled by the user out of all of them
						for (short i = 0; i < HWCounters::TOTAL_NUM_EVENTS; ++i) {
							std::string eventDescription(HWCounters::counterDescriptions[i]);
							if (categoryNode.dataExists(eventDescription)) {
								bool converted = false;
								_enabledEvents[i] = categoryNode.getData(eventDescription, converted);
								assert(converted);
							}
						}
					} else {
						FatalErrorHandler::fail(
							"Unexpected '", category, "' label found while processing the ",
							"hardware counters configuration file."
						);
					}
				}
			);
		}
	}

	static inline void checkIncompatibleBackends()
	{
		if (_enabled[HWCounters::PAPI_BACKEND] && _enabled[HWCounters::PQOS_BACKEND]) {
			FatalErrorHandler::fail("PAPI and PQoS are incompatible hardware counter libraries");
		}
	}

public:

	//! \brief Initialize the hardware counters API with the correct backend
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

	//! \brief Get a vector of enabled events, where the index is an event type
	//! (HWCounters::counters_t) and the boolean tells wether it is enabled
	static inline const std::vector<bool> &getEnabledCounters()
	{
		return _enabledEvents;
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

	//! \brief Start reading hardware counters for a task
	//!
	//! \param[out] task The task to start hardware counter monitoring for
	static void taskStarted(Task *task);

	//! \brief Stop reading hardware counters for a task
	//!
	//! \param[out] task The task to stop hardware counters monitoring for
	static void taskStopped(Task *task);

	//! \brief Finish monitoring a task's hardware counters and accumulate them
	//!
	//! \param[out] task The task to finish hardware counters monitoring for
	static void taskFinished(Task *task);

};

#endif // HARDWARE_COUNTERS_HPP
