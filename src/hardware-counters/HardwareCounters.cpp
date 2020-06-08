/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "HardwareCounters.hpp"
#include "CPUHardwareCounters.hpp"
#include "CPUHardwareCountersInterface.hpp"
#include "TaskHardwareCounters.hpp"
#include "TaskHardwareCountersInterface.hpp"
#include "ThreadHardwareCounters.hpp"
#include "ThreadHardwareCountersInterface.hpp"
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


EnvironmentVariable<bool> HardwareCounters::_verbose("NANOS6_HWCOUNTERS_VERBOSE", false);
EnvironmentVariable<std::string> HardwareCounters::_verboseFile("NANOS6_HWCOUNTERS_VERBOSE_FILE", "nanos6-output-hwcounters.txt");
HardwareCountersInterface *HardwareCounters::_papiBackend;
HardwareCountersInterface *HardwareCounters::_pqosBackend;
HardwareCountersInterface *HardwareCounters::_raplBackend;
bool HardwareCounters::_anyBackendEnabled(false);
std::vector<bool> HardwareCounters::_enabled(HWCounters::NUM_BACKENDS);
std::vector<HWCounters::counters_t> HardwareCounters::_enabledEvents;


void HardwareCounters::loadConfigurationFile()
{
	JsonFile configFile = JsonFile("./nanos6_hwcounters.json");
	if (configFile.fileExists()) {
		configFile.loadData();

		// Navigate through the file and extract the enabled backens and counters
		configFile.getRootNode()->traverseChildrenNodes(
			[&](const std::string &backend, const JsonNode<> &backendNode) {
				if (backend == "PAPI") {
					if (backendNode.dataExists("ENABLED")) {
						bool converted = false;
						bool enabled = backendNode.getData("ENABLED", converted);
						assert(converted);

						_enabled[HWCounters::PAPI_BACKEND] = enabled;
						if (enabled) {
							for (short i = HWCounters::PAPI_MIN_EVENT; i <= HWCounters::PAPI_MAX_EVENT; ++i) {
								std::string eventDescription(HWCounters::counterDescriptions[i]);
								if (backendNode.dataExists(eventDescription)) {
									converted = false;
									if (backendNode.getData(eventDescription, converted) == 1) {
										_enabledEvents.push_back((HWCounters::counters_t) i);
									}
									assert(converted);
								}
							}
						}
					}
				} else if (backend == "PQOS") {
					if (backendNode.dataExists("ENABLED")) {
						bool converted = false;
						bool enabled = backendNode.getData("ENABLED", converted);
						assert(converted);

						_enabled[HWCounters::PQOS_BACKEND] = enabled;
						if (enabled) {
							for (short i = HWCounters::PQOS_MIN_EVENT; i <= HWCounters::PQOS_MAX_EVENT; ++i) {
								std::string eventDescription(HWCounters::counterDescriptions[i]);
								if (backendNode.dataExists(eventDescription)) {
									converted = false;
									if (backendNode.getData(eventDescription, converted) == 1) {
										_enabledEvents.push_back((HWCounters::counters_t) i);
									}
									assert(converted);
								}
							}
						}
					}
				} else if (backend == "RAPL") {
					if (backendNode.dataExists("ENABLED")) {
						bool converted = false;
						bool enabled = backendNode.getData("ENABLED", converted);
						assert(converted);

						_enabled[HWCounters::RAPL_BACKEND] = enabled;
					}
				} else {
					FatalErrorHandler::fail(
						"Unexpected '", backend, "' backend name found while processing the ",
						"hardware counters configuration file."
					);
				}
			}
		);
	}
}

void HardwareCounters::preinitialize()
{
	// First set all backends to nullptr and all events disabled
	_pqosBackend = nullptr;
	_papiBackend = nullptr;
	_raplBackend = nullptr;
	for (short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
		_enabled[i] = false;
	}

	// Load the configuration file to check which backends and events are enabled
	loadConfigurationFile();

	// Check if there's an incompatibility between backends
	checkIncompatibleBackends();

	for (unsigned short i = 0; i < HWCounters::NUM_BACKENDS; ++i) {
		if (_enabled[i]) {
			_anyBackendEnabled = true;
			break;
		}
	}

	// Check which backends must be initialized
	if (_enabled[HWCounters::PQOS_BACKEND]) {
#if HAVE_PQOS
		_pqosBackend = new PQoSHardwareCounters(
			_verbose.getValue(),
			_verboseFile.getValue(),
			_enabledEvents
		);
#else
		FatalErrorHandler::warn("PQoS library not found, disabling hardware counters.");
		_enabled[HWCounters::PQOS_BACKEND] = false;
#endif
	}

	if (_enabled[HWCounters::PAPI_BACKEND]) {
#if HAVE_PAPI
		_papiBackend = new PAPIHardwareCounters(
			_verbose.getValue(),
			_verboseFile.getValue(),
			_enabledEvents
		);
#else
		FatalErrorHandler::warn("PAPI library not found, disabling hardware counters.");
		_enabled[HWCounters::PAPI_BACKEND] = false;
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

	if (_enabled[HWCounters::RAPL_BACKEND]) {
		assert(_raplBackend != nullptr);

		delete _raplBackend;
		_raplBackend = nullptr;
		_enabled[HWCounters::RAPL_BACKEND] = false;
	}

	_anyBackendEnabled = false;
}

void HardwareCounters::cpuBecomesIdle()
{
	WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
	assert(thread != nullptr);

	CPU *cpu = thread->getComputePlace();
	assert(cpu != nullptr);

	CPUHardwareCounters &cpuCounters = cpu->getHardwareCounters();
	ThreadHardwareCounters &threadCounters = thread->getHardwareCounters();
	if (_enabled[HWCounters::PQOS_BACKEND]) {
		assert(_pqosBackend != nullptr);

		_pqosBackend->cpuBecomesIdle(cpuCounters.getPQoSCounters(), threadCounters.getPQoSCounters());
	}

	if (_enabled[HWCounters::PAPI_BACKEND]) {
		assert(_pqosBackend != nullptr);

		_papiBackend->cpuBecomesIdle(cpuCounters.getPAPICounters(), threadCounters.getPAPICounters());
	}
}

void HardwareCounters::threadInitialized()
{
	WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
	assert(thread != nullptr);

	// After the thread is created, initialize (construct) hardware counters
	ThreadHardwareCounters &threadCounters = thread->getHardwareCounters();
	threadCounters.initialize();
	if (_enabled[HWCounters::PQOS_BACKEND]) {
		assert(_pqosBackend != nullptr);

		_pqosBackend->threadInitialized(threadCounters.getPQoSCounters());
	}

	if (_enabled[HWCounters::PAPI_BACKEND]) {
		assert(_papiBackend != nullptr);

		_papiBackend->threadInitialized(threadCounters.getPAPICounters());
	}
}

void HardwareCounters::threadShutdown()
{
	WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
	assert(thread != nullptr);

	ThreadHardwareCounters &threadCounters = thread->getHardwareCounters();
	if (_enabled[HWCounters::PQOS_BACKEND]) {
		assert(_pqosBackend != nullptr);

		_pqosBackend->threadShutdown(threadCounters.getPQoSCounters());
	}

	if (_enabled[HWCounters::PAPI_BACKEND]) {
		assert(_papiBackend != nullptr);

		_papiBackend->threadShutdown(threadCounters.getPAPICounters());
	}

	threadCounters.shutdown();
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
		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->taskReinitialized(taskCounters.getPQoSCounters());
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->taskReinitialized(taskCounters.getPAPICounters());
		}
	}
}

void HardwareCounters::taskStarted(Task *task)
{
	if (_anyBackendEnabled) {
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);
		assert(task != nullptr);

		CPU *cpu = thread->getComputePlace();
		assert(cpu != nullptr);

		CPUHardwareCounters &cpuCounters = cpu->getHardwareCounters();
		ThreadHardwareCounters &threadCounters = thread->getHardwareCounters();
		TaskHardwareCounters &taskCounters = task->getHardwareCounters();
		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->taskStarted(
				cpuCounters.getPQoSCounters(),
				threadCounters.getPQoSCounters(),
				taskCounters.getPQoSCounters()
			);
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->taskStarted(
				cpuCounters.getPQoSCounters(),
				threadCounters.getPAPICounters(),
				taskCounters.getPAPICounters()
			);
		}
	}
}

void HardwareCounters::taskStopped(Task *task)
{
	if (_anyBackendEnabled) {
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);
		assert(task != nullptr);

		CPU *cpu = thread->getComputePlace();
		assert(cpu != nullptr);

		CPUHardwareCounters &cpuCounters = cpu->getHardwareCounters();
		ThreadHardwareCounters &threadCounters = thread->getHardwareCounters();
		TaskHardwareCounters &taskCounters = task->getHardwareCounters();
		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->taskStopped(
				cpuCounters.getPQoSCounters(),
				threadCounters.getPQoSCounters(),
				taskCounters.getPQoSCounters()
			);
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->taskStopped(
				cpuCounters.getPAPICounters(),
				threadCounters.getPAPICounters(),
				taskCounters.getPAPICounters()
			);
		}
	}
}

void HardwareCounters::taskFinished(Task *task)
{
	if (_anyBackendEnabled) {
		assert(task != nullptr);

		TaskHardwareCounters &taskCounters = task->getHardwareCounters();
		if (_enabled[HWCounters::PQOS_BACKEND]) {
			assert(_pqosBackend != nullptr);

			_pqosBackend->taskFinished(task, taskCounters.getPQoSCounters());
		}

		if (_enabled[HWCounters::PAPI_BACKEND]) {
			assert(_papiBackend != nullptr);

			_papiBackend->taskFinished(task, taskCounters.getPAPICounters());
		}

		// Destroy objects for tasks that are not taskfor collaborators; taskfor
		// collaborators reinitialize their structures
		if (!task->isTaskforCollaborator()) {
			taskCounters.shutdown();
		}
	}
}
