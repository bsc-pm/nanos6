/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "CPUHardwareCounters.hpp"
#include "HardwareCounters.hpp"
#include "TaskHardwareCounters.hpp"
#include "ThreadHardwareCounters.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware-counters/rapl/RAPLHardwareCounters.hpp"
#include "support/JsonFile.hpp"
#include "tasks/Task.hpp"

#if HAVE_PAPI
#include "hardware-counters/papi/PAPIHardwareCounters.hpp"
#endif

#if HAVE_PQOS
#include "hardware-counters/pqos/PQoSHardwareCounters.hpp"
#endif


ConfigVariable<bool> HardwareCounters::_verbose("hardware_counters.verbose", false);
ConfigVariable<std::string> HardwareCounters::_verboseFile("hardware_counters.verbose_file", "nanos6-output-hwcounters.txt");
HardwareCountersInterface *HardwareCounters::_papiBackend;
HardwareCountersInterface *HardwareCounters::_pqosBackend;
HardwareCountersInterface *HardwareCounters::_raplBackend;
bool HardwareCounters::_anyBackendEnabled(false);
std::vector<bool> HardwareCounters::_enabled(HWCounters::NUM_BACKENDS);
std::vector<HWCounters::counters_t> HardwareCounters::_enabledCounters;


void HardwareCounters::loadConfiguration()
{
	ConfigVariable<bool> papiEnabled("hardware_counters.papi.enabled", false);
	ConfigVariable<bool> pqosEnabled("hardware_counters.pqos.enabled", false);
	ConfigVariable<bool> raplEnabled("hardware_counters.rapl.enabled", false);

	_enabled[HWCounters::PAPI_BACKEND] = papiEnabled;
	if (papiEnabled) {
		_anyBackendEnabled = true;
		ConfigVariableSet<std::string> counterSet("hardware_counters.papi.counters", {});

		for (short i = HWCounters::HWC_PAPI_MIN_EVENT; i <= HWCounters::HWC_PAPI_MAX_EVENT; ++i) {
			std::string eventDescription(HWCounters::counterDescriptions[i]);
			if (counterSet.contains(eventDescription))
				_enabledCounters.push_back((HWCounters::counters_t) i);
		}
	}

	_enabled[HWCounters::PQOS_BACKEND] = pqosEnabled;
	if (pqosEnabled) {
		_anyBackendEnabled = true;
		ConfigVariableSet<std::string> counterSet("hardware_counters.pqos.counters", {});

		for (short i = HWCounters::HWC_PQOS_MIN_EVENT; i <= HWCounters::HWC_PQOS_MAX_EVENT; ++i) {
			std::string eventDescription(HWCounters::counterDescriptions[i]);
			if (counterSet.contains(eventDescription))
				_enabledCounters.push_back((HWCounters::counters_t) i);
		}
	}
}

void HardwareCounters::preinitialize()
{
	// First set all backends to nullptr and all events disabled
	_papiBackend = nullptr;
	_pqosBackend = nullptr;
	_raplBackend = nullptr;
	for (short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
		_enabled[i] = false;
	}

	// Load the configuration file to check which backends and events are enabled
	loadConfiguration();

	// Check if there's an incompatibility between backends
	checkIncompatibilities();

	// If verbose is enabled and no backends are available, warn the user
	if (!_anyBackendEnabled && _verbose.getValue()) {
		FatalErrorHandler::warn("Hardware Counters verbose mode enabled but no backends available!");
	}

	if (_enabled[HWCounters::PAPI_BACKEND]) {
#if HAVE_PAPI
		_papiBackend = new PAPIHardwareCounters(
			_verbose.getValue(),
			_verboseFile.getValue(),
			_enabledCounters
		);
#else
		FatalErrorHandler::warn("PAPI library not found, disabling hardware counters.");
		_enabled[HWCounters::PAPI_BACKEND] = false;
#endif
	}

	// Check which backends must be initialized
	if (_enabled[HWCounters::PQOS_BACKEND]) {
#if HAVE_PQOS
		_pqosBackend = new PQoSHardwareCounters(
			_verbose.getValue(),
			_verboseFile.getValue(),
			_enabledCounters
		);
#else
		FatalErrorHandler::warn("PQoS library not found, disabling hardware counters.");
		_enabled[HWCounters::PQOS_BACKEND] = false;
#endif
	}

	// NOTE: Since the RAPL backend needs to be initialized after hardware is
	// detected, we do that in the initialize function
}

void HardwareCounters::initialize()
{
	if (_enabled[HWCounters::RAPL_BACKEND]) {
		_raplBackend = new RAPLHardwareCounters(
			_verbose.getValue(),
			_verboseFile.getValue()
		);
	}
}

void HardwareCounters::shutdown()
{
	if (_enabled[HWCounters::PAPI_BACKEND]) {
		assert(_papiBackend != nullptr);

		delete _papiBackend;
		_papiBackend = nullptr;
		_enabled[HWCounters::PAPI_BACKEND] = false;
	}

	if (_enabled[HWCounters::PQOS_BACKEND]) {
		assert(_pqosBackend != nullptr);

		delete _pqosBackend;
		_pqosBackend = nullptr;
		_enabled[HWCounters::PQOS_BACKEND] = false;
	}

	if (_enabled[HWCounters::RAPL_BACKEND]) {
		assert(_raplBackend != nullptr);

		delete _raplBackend;
		_raplBackend = nullptr;
		_enabled[HWCounters::RAPL_BACKEND] = false;
	}

	_anyBackendEnabled = false;
}

void HardwareCounters::threadInitialized()
{
	WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
	assert(thread != nullptr);

	// After the thread is created, initialize (construct) hardware counters
	ThreadHardwareCounters &threadCounters = thread->getHardwareCounters();
	threadCounters.initialize();
	if (_enabled[HWCounters::PAPI_BACKEND]) {
		assert(_papiBackend != nullptr);

		_papiBackend->threadInitialized(threadCounters.getPAPICounters());
	}

	if (_enabled[HWCounters::PQOS_BACKEND]) {
		assert(_pqosBackend != nullptr);

		_pqosBackend->threadInitialized(threadCounters.getPQoSCounters());
	}
}

void HardwareCounters::threadShutdown()
{
	WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
	assert(thread != nullptr);

	ThreadHardwareCounters &threadCounters = thread->getHardwareCounters();
	if (_enabled[HWCounters::PAPI_BACKEND]) {
		assert(_papiBackend != nullptr);

		_papiBackend->threadShutdown(threadCounters.getPAPICounters());
	}

	if (_enabled[HWCounters::PQOS_BACKEND]) {
		assert(_pqosBackend != nullptr);

		_pqosBackend->threadShutdown(threadCounters.getPQoSCounters());
	}
}

void HardwareCounters::taskCreated(Task *task, bool enabled)
{
	if (_anyBackendEnabled) {
		assert(task != nullptr);

		// After the task is created, initialize (construct) hardware counters
		TaskHardwareCounters &taskCounters = task->getHardwareCounters();
		taskCounters.initialize(enabled);
	}
}

void HardwareCounters::taskReinitialized(Task *task)
{
	if (_anyBackendEnabled) {
		assert(task != nullptr);

		TaskHardwareCounters &taskCounters = task->getHardwareCounters();
		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->taskReinitialized(taskCounters.getPAPICounters());
		}

		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->taskReinitialized(taskCounters.getPQoSCounters());
		}
	}
}

void HardwareCounters::updateTaskCounters(Task *task)
{
	if (_anyBackendEnabled) {
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);
		assert(task != nullptr);

		ThreadHardwareCounters &threadCounters = thread->getHardwareCounters();
		TaskHardwareCounters &taskCounters = task->getHardwareCounters();
		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->updateTaskCounters(
				threadCounters.getPAPICounters(),
				taskCounters.getPAPICounters()
			);
		}

		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->updateTaskCounters(
				threadCounters.getPQoSCounters(),
				taskCounters.getPQoSCounters()
			);
		}
	}
}

void HardwareCounters::updateRuntimeCounters()
{
	if (_anyBackendEnabled) {
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);

		CPU *cpu = thread->getComputePlace();
		assert(cpu != nullptr);

		CPUHardwareCounters &cpuCounters = cpu->getHardwareCounters();
		ThreadHardwareCounters &threadCounters = thread->getHardwareCounters();
		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->updateRuntimeCounters(
				cpuCounters.getPAPICounters(),
				threadCounters.getPAPICounters()
			);
		}

		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->updateRuntimeCounters(
				cpuCounters.getPQoSCounters(),
				threadCounters.getPQoSCounters()
			);
		}
	}
}
