/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <fstream>

#include "HardwareCounters.hpp"
#include "executors/threads/WorkerThread.hpp"


EnvironmentVariable<bool> HardwareCounters::_enabled("NANOS6_HARDWARE_COUNTERS_ENABLE", false);
EnvironmentVariable<bool> HardwareCounters::_verbose("NANOS6_HARDWARE_COUNTERS_VERBOSE", false);
EnvironmentVariable<std::string> HardwareCounters::_outputFile("NANOS6_HARDWARE_COUNTERS_VERBOSE_FILE", "output-hardware-counters.txt");
HardwareCounters* HardwareCounters::_monitor;
enum pqos_mon_event _monitoredEvents;


//    HARDWARE COUNTERS    //

void HardwareCounters::initialize()
{
	if (_enabled) {
		// Create the hardware counters monitor module
		if (_monitor == nullptr) {
			_monitor = new HardwareCounters();
		}
		
		// Declare PQoS configuration and capabilities structures
		pqos_config configuration;
		const pqos_cpuinfo *pqosCPUInfo                   = NULL;
		const pqos_cap *pqosCapabilities                  = NULL;
		const pqos_capability *pqosMonitoringCapabilities = NULL;
		
		// Get the configuration features
		memset(&configuration, 0, sizeof(configuration));
		configuration.fd_log    = STDOUT_FILENO;
		configuration.verbose   = 0;
		configuration.interface = PQOS_INTER_OS;
		
		// Check and initialize PQoS CMT capabilities
		int ret = pqos_init(&configuration);
		FatalErrorHandler::failIf(ret != PQOS_RETVAL_OK, "Error '", ret, "' when initializing the PQoS library");
		
		// Get PQoS CMT capabilities and CPU info pointer
		ret = pqos_cap_get(&pqosCapabilities, &pqosCPUInfo);
		FatalErrorHandler::failIf(ret != PQOS_RETVAL_OK, "Error '", ret, "' when retrieving PQoS capabilities");
		ret = pqos_cap_get_type(pqosCapabilities, PQOS_CAP_TYPE_MON, &pqosMonitoringCapabilities);
		FatalErrorHandler::failIf(ret != PQOS_RETVAL_OK, "Error '", ret, "' when retrieving PQoS capability types");
		
		assert(pqosCapabilities != nullptr);
		assert(pqosCPUInfo != nullptr);
		
		// Choose events to monitor:
		//  IPC (Instructions retired / cycles)
		//    - Computed using the number of instructions and cycles executed
		//      from the initialization of the library until a snapshot is taken
		//  LLC Usage (L3 Occupancy)
		//    - Similar to IPC, reports llc usage at the moment of snapshot taking
		//  MEM BW
		//    - Local memory bandwidth
		//    - Remote memory bandwidth
		//  LLC Misses
		//    - Number of LLC misses
		const enum pqos_mon_event eventsToMonitor = (pqos_mon_event)
		(
			PQOS_PERF_EVENT_IPC      | // IPC
			PQOS_PERF_EVENT_LLC_MISS | // LLC Misses
			PQOS_MON_EVENT_LMEM_BW   | // Local Memory Bandwidth
			PQOS_MON_EVENT_RMEM_BW   | // Remote Memory Bandwidth
			PQOS_MON_EVENT_L3_OCCUP    // LLC Usage
		);
		
		assert(pqosMonitoringCapabilities->u.mon != nullptr);
		
		// Check available events
		enum pqos_mon_event availableEvents = (pqos_mon_event) 0;
		for (unsigned int i = 0; i < pqosMonitoringCapabilities->u.mon->num_events; i++) {
			availableEvents = (pqos_mon_event) (availableEvents | (pqosMonitoringCapabilities->u.mon->events[i].type));
		}
		
		// Only choose events we want to monitor that are available
		_monitoredEvents = (pqos_mon_event) (availableEvents & eventsToMonitor);
		
		// If none of the events can be monitored, trigger an early shutdown
		if (_monitoredEvents == ((pqos_mon_event) 0)) {
			shutdown();
		}
		
		// Propagate initialization to thread and task Hardware counter monitors
		TaskHardwareCountersMonitor::initialize();
		ThreadHardwareCountersMonitor::initialize();
	}
}

void HardwareCounters::shutdown()
{
	if (_enabled) {
		// Shutdown PQoS monitoring
		int ret = pqos_fini();
		FatalErrorHandler::failIf(ret != PQOS_RETVAL_OK, "Error '", ret, "' when shutting down the PQoS library");
		
		// Display monitoring statistics
		displayStatistics();
		
		// Propagate shutdown to thread and task Hardware counter monitors
		TaskHardwareCountersMonitor::shutdown();
		ThreadHardwareCountersMonitor::shutdown();
		
		// Destroy the monitoring module
		if (_monitor != nullptr) {
			delete _monitor;
		}
		
		_enabled.setValue(false);
	}
}

void HardwareCounters::displayStatistics()
{
	if (_enabled && _verbose) {
		// Try opening the output file
		std::ios_base::openmode openMode = std::ios::out;
		std::ofstream output(_outputFile.getValue(), openMode);
		FatalErrorHandler::warnIf(
			!output.is_open(),
			"Could not create or open the verbose file: ",
			_outputFile.getValue(),
			". Using standard output."
		);
		
		// Retrieve statistics from every module
		std::stringstream outputStream;
		TaskHardwareCountersMonitor::displayStatistics(outputStream);
		ThreadHardwareCountersMonitor::displayStatistics(outputStream);
		
		if (output.is_open()) {
			// Output into the file and close it
			output << outputStream.str();
			output.close();
		}
		else {
			std::cout << outputStream.str();
		}
	}
}

bool HardwareCounters::isEnabled()
{
	return _enabled;
}


//    TASKS    //

void HardwareCounters::taskCreated(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);
		
		// Retrieve information about the task
		TaskHardwareCounters *taskCounters               = task->getTaskHardwareCounters();
		TaskHardwareCountersPredictions *taskPredictions = task->getTaskHardwareCountersPredictions();
		const std::string &label = task->getLabel();
		size_t cost = (task->hasCost() ? task->getCost() : DEFAULT_COST);
		
		// Create task hardware counter structures and predict counter values
		TaskHardwareCountersMonitor::taskCreated(taskCounters, label, cost);
		TaskHardwareCountersMonitor::predictTaskCounters(taskPredictions, label, cost);
	}
}

void HardwareCounters::startTaskMonitoring(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);
		
		// Get the thread's and task's hardware counter structures
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		if (thread != nullptr) {
			pqos_mon_data *threadData = thread->getThreadHardwareCounters()->getData();
			TaskHardwareCounters *taskCounters = task->getTaskHardwareCounters();
			
			// Start or resume Hardware counter monitoring for the task
			TaskHardwareCountersMonitor::startTaskMonitoring(taskCounters, threadData);
		}
	}
}

void HardwareCounters::stopTaskMonitoring(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);
		
		// Get the thread's and task's hardware counter structures
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		if (thread != nullptr) {
			pqos_mon_data *threadData = thread->getThreadHardwareCounters()->getData();
			TaskHardwareCounters *taskCounters = task->getTaskHardwareCounters();
			
			// Stop or pause Hardware counter monitoring for the task
			TaskHardwareCountersMonitor::stopTaskMonitoring(taskCounters, threadData);
		}
	}
}

void HardwareCounters::taskFinished(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);
		
		// Get the task's hardware counter structures
		TaskHardwareCounters *taskCounters = task->getTaskHardwareCounters();
		TaskHardwareCountersPredictions *taskPredictions = task->getTaskHardwareCountersPredictions();
		
		// Finish Hardware counter monitoring for the task
		TaskHardwareCountersMonitor::taskFinished(taskCounters, taskPredictions);
	}
}


//    THREADS    //

void HardwareCounters::initializeThread()
{
	if (_enabled) {
		// Get the thread's hardware counter structures
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);
		ThreadHardwareCounters *threadCounters = thread->getThreadHardwareCounters();
		
		// Initialize HW counter monitoring for the thread
		ThreadHardwareCountersMonitor::initializeThread(threadCounters, _monitoredEvents);
	}
}

void HardwareCounters::shutdownThread()
{
	if (_enabled) {
		// Get the thread's hardware counter structures
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);
		ThreadHardwareCounters *threadCounters = thread->getThreadHardwareCounters();
		
		// Shutdown hw counter monitoring for the thread
		ThreadHardwareCountersMonitor::shutdownThread(threadCounters);
	}
}
