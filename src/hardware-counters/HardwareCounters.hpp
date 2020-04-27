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

#if HAVE_PAPI
#include "hardware-counters/papi/PAPIHardwareCounters.hpp"
#endif

#if HAVE_PQOS
#include "hardware-counters/pqos/PQoSHardwareCounters.hpp"
#endif


class Task;

class HardwareCounters {

private:

	//! An env var that shows the chosen backends (defaults to null)
	static EnvironmentVariable<std::string> _chosenBackend;

	//! The underlying backends
	static std::vector<HardwareCountersInterface *> _backends;

	//! Whether each backend is enabled
	static std::vector<bool> _enabled;

	//! Whether the verbose mode is enabled
	static EnvironmentVariable<bool> _verbose;

	//! The file where output must be saved when verbose mode is enabled
	static EnvironmentVariable<std::string> _verboseFile;

public:

	//! \brief Initialize the hardware counters API with the correct backend
	static inline void initialize()
	{
		for (short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
			_backends[i] = nullptr;
			_enabled[i] = false;
		}

		// Check which backends must be initialized
		if (_chosenBackend.getValue() == "papi") {
#if HAVE_PAPI
			_backends[HWCounters::PAPI_BACKEND] = new PAPIHardwareCounters(_verbose.getValue(), _verboseFile.getValue());
			_enabled[HWCounters::PAPI_BACKEND] = true;
#else
			FatalErrorHandler::warn("PAPI library not found, disabling hardware counters.");
#endif
		}

		if (_chosenBackend.getValue() == "pqos") {
#if HAVE_PQOS
			_backends[HWCounters::PQOS_BACKEND] = new PQoSHardwareCounters(_verbose.getValue(), _verboseFile.getValue());
			_enabled[HWCounters::PQOS_BACKEND] = true;
#else
			FatalErrorHandler::warn("PQoS library not found, disabling hardware counters.");
#endif
		}

		if (_chosenBackend.getValue() != "null" &&
			_chosenBackend.getValue() != "papi" &&
			_chosenBackend.getValue() != "pqos"
		) {
			FatalErrorHandler::fail(
				"Unexistent backend for hardware counters instrumentation: ",
				_chosenBackend.getValue()
			);
		}
	}

	//! \brief Shutdown the hardware counters API
	static inline void shutdown()
	{
		for (short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
			if (_enabled[i]) {
				assert(_backends[i] != nullptr);

				delete _backends[i];
				_backends[i] = nullptr;
				_enabled[i] = false;
			}
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

		ThreadHardwareCounters *hardwareCounters = thread->getHardwareCounters();
		assert(hardwareCounters != nullptr);

		// After the thread is created, initialize (construct) hardware counters
		hardwareCounters->initialize();

		for (short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
			if (_enabled[i]) {
				assert(_backends[i] != nullptr);

				_backends[i]->threadInitialized();
			}
		}
	}

	//! \brief Destroy the hardware counter structures of a thread
	static inline void threadShutdown()
	{
		for (short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
			if (_enabled[i]) {
				assert(_backends[i] != nullptr);

				_backends[i]->threadShutdown();
			}
		}
	}

	//! \brief Initialize hardware counter structures for a task
	//! \param[out] task The task to create structures for
	//! \param[in] enabled Whether to create structures and monitor this task
	static inline void taskCreated(Task *task, bool enabled = true)
	{
		assert(task != nullptr);

		TaskHardwareCounters *hardwareCounters = task->getHardwareCounters();
		assert(hardwareCounters != nullptr);

		// After the task is created, initialize (construct) hardware counters
		hardwareCounters->initialize();

		for (short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
			if (_enabled[i]) {
				assert(_backends[i] != nullptr);

				_backends[i]->taskCreated(task, enabled);
			}
		}
	}

	//! \brief Reinitialize all hardware counter structures for a task
	//! \param[out] task The task to reinitialize structures for
	static inline void taskReinitialized(Task *task)
	{
		for (short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
			if (_enabled[i]) {
				assert(_backends[i] != nullptr);

				_backends[i]->taskReinitialized(task);
			}
		}
	}

	//! \brief Start reading hardware counters for a task
	//! \param[out] task The task to start hardware counter monitoring for
	static inline void taskStarted(Task *task)
	{
		for (short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
			if (_enabled[i]) {
				assert(_backends[i] != nullptr);

				_backends[i]->taskStarted(task);
			}
		}
	}

	//! \brief Stop reading hardware counters for a task
	//! \param[out] task The task to stop hardware counters monitoring for
	static inline void taskStopped(Task *task)
	{
		for (short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
			if (_enabled[i]) {
				assert(_backends[i] != nullptr);

				_backends[i]->taskStopped(task);
			}
		}
	}

	//! \brief Finish monitoring a task's hardware counters and accumulate them
	//! \param[out] task The task to finish hardware counters monitoring for
	static inline void taskFinished(Task *task)
	{
		for (short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
			if (_enabled[i]) {
				assert(_backends[i] != nullptr);

				_backends[i]->taskFinished(task);
			}
		}
	}

};

#endif // HARDWARE_COUNTERS_HPP
