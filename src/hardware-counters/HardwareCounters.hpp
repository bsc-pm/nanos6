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

public:

	//! \brief Initialize the hardware counters API with the correct backend
	static inline void initialize()
	{
		// First set all backends to nullptr
		_pqosBackend = nullptr;
		_papiBackend = nullptr;
		for (short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
			_enabled[i] = false;
		}

		// Check which backends must be initialized
		if (_chosenBackend.getValue() == "papi") {
#if HAVE_PAPI
			_papiBackend = (HardwareCountersInterface *) new PAPIHardwareCounters(_verbose.getValue(), _verboseFile.getValue());
			_enabled[HWCounters::PAPI_BACKEND] = true;
#else
			FatalErrorHandler::warn("PAPI library not found, disabling hardware counters.");
#endif
		}

		if (_chosenBackend.getValue() == "pqos") {
#if HAVE_PQOS
			_pqosBackend = (HardwareCountersInterface *) new PQoSHardwareCounters(_verbose.getValue(), _verboseFile.getValue());
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

		ThreadHardwareCounters *hardwareCounters = thread->getHardwareCounters();
		assert(hardwareCounters != nullptr);

		// After the thread is created, initialize (construct) hardware counters
		hardwareCounters->initialize();

		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->threadInitialized();
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->threadInitialized();
		}
	}

	//! \brief Destroy the hardware counter structures of a thread
	static inline void threadShutdown()
	{
		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->threadShutdown();
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->threadShutdown();
		}
	}

	//! \brief Initialize hardware counter structures for a task
	//!
	//! \param[out] task The task to create structures for
	//! \param[in] enabled Whether to create structures and monitor this task
	static inline void taskCreated(Task *task, bool enabled = true)
	{
		assert(task != nullptr);

		TaskHardwareCounters *hardwareCounters = task->getHardwareCounters();
		assert(hardwareCounters != nullptr);

		// After the task is created, initialize (construct) hardware counters
		hardwareCounters->initialize();

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
		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->taskReinitialized(task);
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->taskReinitialized(task);
		}
	}

	//! \brief Start reading hardware counters for a task
	//!
	//! \param[out] task The task to start hardware counter monitoring for
	static inline void taskStarted(Task *task)
	{
		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->taskStarted(task);
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->taskStarted(task);
		}
	}

	//! \brief Stop reading hardware counters for a task
	//!
	//! \param[out] task The task to stop hardware counters monitoring for
	static inline void taskStopped(Task *task)
	{
		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->taskStopped(task);
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->taskStopped(task);
		}
	}

	//! \brief Finish monitoring a task's hardware counters and accumulate them
	//!
	//! \param[out] task The task to finish hardware counters monitoring for
	static inline void taskFinished(Task *task)
	{
		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->taskFinished(task);
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->taskFinished(task);
		}
	}

};

#endif // HARDWARE_COUNTERS_HPP
