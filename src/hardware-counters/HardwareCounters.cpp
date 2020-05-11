/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "HardwareCounters.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware-counters/TaskHardwareCounters.hpp"
#include "hardware-counters/TaskHardwareCountersInterface.hpp"
#include "hardware-counters/ThreadHardwareCounters.hpp"
#include "hardware-counters/ThreadHardwareCountersInterface.hpp"
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
std::vector<bool> HardwareCounters::_enabled(HWCounters::NUM_BACKENDS);
std::vector<bool> HardwareCounters::_enabledEvents(HWCounters::TOTAL_NUM_EVENTS);


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
									_enabledEvents[i] = backendNode.getData(eventDescription, converted);
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
									_enabledEvents[i] = backendNode.getData(eventDescription, converted);
									assert(converted);
								}
							}
						}
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

void HardwareCounters::initialize()
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

	// Check if there's an incompatibility between backends
	checkIncompatibleBackends();

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
		_enabled[HWCounters::PAPI_BACKEND] = false;
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
		_enabled[HWCounters::PQOS_BACKEND] = false;
#endif
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
}

void HardwareCounters::threadInitialized()
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

void HardwareCounters::threadShutdown()
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

void HardwareCounters::taskCreated(Task *task, bool enabled)
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

void HardwareCounters::taskReinitialized(Task *task)
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

void HardwareCounters::taskStarted(Task *task)
{
	assert(task != nullptr);

	if (_enabled[HWCounters::PQOS_BACKEND]) {
		assert(_pqosBackend != nullptr);

		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);

		ThreadHardwareCounters *threadCounters = thread->getHardwareCounters();
		TaskHardwareCounters *taskCounters = task->getHardwareCounters();
		assert(threadCounters != nullptr);
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
		TaskHardwareCounters *taskCounters = task->getHardwareCounters();
		assert(threadCounters != nullptr);
		assert(taskCounters != nullptr);

		ThreadHardwareCountersInterface *papiThreadCounters = threadCounters->getPAPICounters();
		TaskHardwareCountersInterface *papiTaskCounters = taskCounters->getPAPICounters();
		_papiBackend->taskStarted(papiThreadCounters, papiTaskCounters);
	}
}

void HardwareCounters::taskStopped(Task *task)
{
	assert(task != nullptr);

	if (_enabled[HWCounters::PQOS_BACKEND]) {
		assert(_pqosBackend != nullptr);

		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);

		ThreadHardwareCounters *threadCounters = thread->getHardwareCounters();
		TaskHardwareCounters *taskCounters = task->getHardwareCounters();
		assert(threadCounters != nullptr);
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
		TaskHardwareCounters *taskCounters = task->getHardwareCounters();
		assert(threadCounters != nullptr);
		assert(taskCounters != nullptr);

		ThreadHardwareCountersInterface *papiThreadCounters = threadCounters->getPAPICounters();
		TaskHardwareCountersInterface *papiTaskCounters = taskCounters->getPAPICounters();
		_papiBackend->taskStopped(papiThreadCounters, papiTaskCounters);
	}
}

void HardwareCounters::taskFinished(Task *task)
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
