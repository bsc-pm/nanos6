/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef HARDWARE_COUNTERS_HPP
#define HARDWARE_COUNTERS_HPP

#include <vector>

#include "HardwareCountersInterface.hpp"
#include "SupportedHardwareCounters.hpp"
#include "hardware-counters/TaskHardwareCounters.hpp"
#include "hardware-counters/TaskHardwareCountersInterface.hpp"
#include "hardware-counters/ThreadHardwareCounters.hpp"
#include "hardware-counters/ThreadHardwareCountersInterface.hpp"
#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "support/JsonFile.hpp"

#if HAVE_PAPI
#include "hardware-counters/papi/PAPIHardwareCounters.hpp"
#endif

#if HAVE_PQOS
#include "hardware-counters/pqos/PQoSHardwareCounters.hpp"
#endif


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

public:

	//! \brief Initialize the hardware counters API with the correct backend
	static inline void initialize()
	{
		// First set all backends to nullptr and all events disabled
		_pqosBackend = nullptr;
		_papiBackend = nullptr;
		for (short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
			_enabled[i] = false;
		}

		for (short i = 0; i < HWCounters::TOTAL_NUM_EVENTS; ++i) {
			_enabledEvents[i] = false;
		}

		// Load the configuration file to check which backends and events are enabled
		loadConfigurationFile();

		// Check which backends must be initialized
		if (_enabled[HWCounters::PAPI_BACKEND]) {
#if HAVE_PAPI
			_papiBackend = (HardwareCountersInterface *) new PAPIHardwareCounters(
				_verbose.getValue(),
				_verboseFile.getValue(),
				_enabledEvents
			);
#else
			FatalErrorHandler::warn("PAPI library not found, disabling hardware counters.");
#endif
		}

		if (_enabled[HWCounters::PQOS_BACKEND]) {
#if HAVE_PQOS
			_pqosBackend = (HardwareCountersInterface *) new PQoSHardwareCounters(
				_verbose.getValue(),
				_verboseFile.getValue(),
				_enabledEvents
			);
#else
			FatalErrorHandler::warn("PQoS library not found, disabling hardware counters.");
#endif
		}
	}

	//! \brief Shutdown the hardware counters API
	static inline void shutdown()
	{
		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			delete _pqosBackend;
			_pqosBackend = nullptr;
			_enabled[HWCounters::PQOS_BACKEND] = false;
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			delete _papiBackend;
			_papiBackend = nullptr;
			_enabled[HWCounters::PAPI_BACKEND] = false;
		}
	}

	//! \brief Check whether a backend is enabled
	//!
	//! \param[in] backend The backend's id
	static inline bool isEnabled(HWCounters::backends_t backend)
	{
		return _enabled[backend];
	}

	//! \brief Initialize hardware counter structures for a new thread
	static inline void threadInitialized()
	{
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);

		ThreadHardwareCounters *threadCounters = thread->getHardwareCounters();
		assert(threadCounters != nullptr);

		// After the thread is created, initialize (construct) hardware counters
		threadCounters->initialize();

		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			ThreadHardwareCountersInterface *pqosCounters = threadCounters->getPQoSCounters();

			_pqosBackend->threadInitialized(pqosCounters);
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			ThreadHardwareCountersInterface *papiCounters = threadCounters->getPAPICounters();

			_papiBackend->threadInitialized(papiCounters);
		}
	}

	//! \brief Destroy the hardware counter structures of a thread
	static inline void threadShutdown()
	{
		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
			assert(thread != nullptr);

			ThreadHardwareCounters *threadCounters = thread->getHardwareCounters();
			assert(threadCounters != nullptr);

			ThreadHardwareCountersInterface *pqosCounters = threadCounters->getPQoSCounters();

			_pqosBackend->threadShutdown(pqosCounters);
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
			assert(thread != nullptr);

			ThreadHardwareCounters *threadCounters = thread->getHardwareCounters();
			assert(threadCounters != nullptr);

			ThreadHardwareCountersInterface *papiCounters = threadCounters->getPAPICounters();

			_papiBackend->threadShutdown(papiCounters);
		}
	}

	//! \brief Initialize hardware counter structures for a task
	//!
	//! \param[out] task The task to create structures for
	//! \param[in] enabled Whether to create structures and monitor this task
	static inline void taskCreated(Task *task, bool enabled = true)
	{
		assert(task != nullptr);

		TaskHardwareCounters *taskCounters = task->getHardwareCounters();
		assert(taskCounters != nullptr);

		// After the task is created, initialize (construct) hardware counters
		taskCounters->initialize();

		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->taskCreated(task, enabled);
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->taskCreated(task, enabled);
		}
	}

	//! \brief Reinitialize all hardware counter structures for a task
	//!
	//! \param[out] task The task to reinitialize structures for
	static inline void taskReinitialized(Task *task)
	{
		assert(task != nullptr);

		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			TaskHardwareCounters *taskCounters = task->getHardwareCounters();
			assert(taskCounters != nullptr);

			TaskHardwareCountersInterface *pqosTaskCounters = taskCounters->getPQoSCounters();

			_pqosBackend->taskReinitialized(pqosTaskCounters);
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			TaskHardwareCounters *taskCounters = task->getHardwareCounters();
			assert(taskCounters != nullptr);

			TaskHardwareCountersInterface *papiTaskCounters = taskCounters->getPAPICounters();

			_papiBackend->taskReinitialized(papiTaskCounters);
		}
	}

	//! \brief Start reading hardware counters for a task
	//!
	//! \param[out] task The task to start hardware counter monitoring for
	static inline void taskStarted(Task *task)
	{
		assert(task != nullptr);

		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
			assert(thread != nullptr);

			ThreadHardwareCounters *threadCounters = thread->getHardwareCounters();
			assert(threadCounters != nullptr);

			TaskHardwareCounters *taskCounters = task->getHardwareCounters();
			assert(taskCounters != nullptr);

			ThreadHardwareCountersInterface *pqosThreadCounters = threadCounters->getPQoSCounters();
			TaskHardwareCountersInterface *pqosTaskCounters = taskCounters->getPQoSCounters();

			_pqosBackend->taskStarted(pqosThreadCounters, pqosTaskCounters);
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
			assert(thread != nullptr);

			ThreadHardwareCounters *threadCounters = thread->getHardwareCounters();
			assert(threadCounters != nullptr);

			TaskHardwareCounters *taskCounters = task->getHardwareCounters();
			assert(taskCounters != nullptr);

			ThreadHardwareCountersInterface *papiThreadCounters = threadCounters->getPAPICounters();
			TaskHardwareCountersInterface *papiTaskCounters = taskCounters->getPAPICounters();

			_papiBackend->taskStarted(papiThreadCounters, papiTaskCounters);
		}
	}

	//! \brief Stop reading hardware counters for a task
	//!
	//! \param[out] task The task to stop hardware counters monitoring for
	static inline void taskStopped(Task *task)
	{
		assert(task != nullptr);

		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
			assert(thread != nullptr);

			ThreadHardwareCounters *threadCounters = thread->getHardwareCounters();
			assert(threadCounters != nullptr);

			TaskHardwareCounters *taskCounters = task->getHardwareCounters();
			assert(taskCounters != nullptr);

			ThreadHardwareCountersInterface *pqosThreadCounters = threadCounters->getPQoSCounters();
			TaskHardwareCountersInterface *pqosTaskCounters = taskCounters->getPQoSCounters();

			_pqosBackend->taskStopped(pqosThreadCounters, pqosTaskCounters);
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
			assert(thread != nullptr);

			ThreadHardwareCounters *threadCounters = thread->getHardwareCounters();
			assert(threadCounters != nullptr);

			TaskHardwareCounters *taskCounters = task->getHardwareCounters();
			assert(taskCounters != nullptr);

			ThreadHardwareCountersInterface *papiThreadCounters = threadCounters->getPAPICounters();
			TaskHardwareCountersInterface *papiTaskCounters = taskCounters->getPAPICounters();

			_papiBackend->taskStopped(papiThreadCounters, papiTaskCounters);
		}
	}

	//! \brief Finish monitoring a task's hardware counters and accumulate them
	//!
	//! \param[out] task The task to finish hardware counters monitoring for
	static inline void taskFinished(Task *task)
	{
		assert(task != nullptr);

		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			TaskHardwareCounters *taskCounters = task->getHardwareCounters();
			assert(taskCounters != nullptr);

			TaskHardwareCountersInterface *pqosTaskCounters = taskCounters->getPQoSCounters();

			_pqosBackend->taskFinished(task, pqosTaskCounters);
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			TaskHardwareCounters *taskCounters = task->getHardwareCounters();
			assert(taskCounters != nullptr);

			TaskHardwareCountersInterface *papiTaskCounters = taskCounters->getPAPICounters();

			_papiBackend->taskFinished(task, papiTaskCounters);
		}
	}

};

#endif // HARDWARE_COUNTERS_HPP
