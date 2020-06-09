/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <sys/utsname.h>

#include "PQoSHardwareCounters.hpp"
#include "PQoSTaskHardwareCounters.hpp"
#include "PQoSThreadHardwareCounters.hpp"
#include "executors/threads/WorkerThread.hpp"


PQoSHardwareCounters::PQoSHardwareCounters(
	bool,
	const std::string &,
	const std::vector<HWCounters::counters_t> &enabledEvents
) {
	for (unsigned short i = 0; i < HWCounters::PQOS_NUM_EVENTS; ++i) {
		_enabledEvents[i] = false;
	}

	// Check if the PQoS version may give problems
	utsname kernelInfo;
	if (uname(&kernelInfo) == 0) {
		std::string kernelRelease(kernelInfo.release);
		if (kernelRelease.find("4.13.", 0) == 0) {
			if (kernelRelease.find("4.13.0", 0) != 0) {
				FatalErrorHandler::warn("4.13.X (X != 0) kernel versions may give incorrect readings for MBM counters");
			}
		}
	}

	// Declare PQoS configuration and capabilities structures
	pqos_config configuration;
	const pqos_cpuinfo *pqosCPUInfo = nullptr;
	const pqos_cap *pqosCap = nullptr;
	const pqos_capability *pqosCapabilities = nullptr;

	// Get the configuration features
	memset(&configuration, 0, sizeof(configuration));
	configuration.fd_log = STDOUT_FILENO;
	configuration.verbose = 0;
	configuration.interface = PQOS_INTER_OS;

	// Get PQoS CMT capabilities and CPU info pointer
	int ret = pqos_init(&configuration);
	FatalErrorHandler::failIf(
		ret != PQOS_RETVAL_OK,
		ret, " when initializing the PQoS library"
	);
	ret = pqos_cap_get(&pqosCap, &pqosCPUInfo);
	FatalErrorHandler::failIf(
		ret != PQOS_RETVAL_OK,
		ret, " when retrieving PQoS capabilities"
	);
	ret = pqos_cap_get_type(pqosCap, PQOS_CAP_TYPE_MON, &pqosCapabilities);
	FatalErrorHandler::failIf(
		ret != PQOS_RETVAL_OK,
		ret, " when retrieving PQoS capability types"
	);

	assert(pqosCapabilities != nullptr);
	assert(pqosCapabilities->u.mon != nullptr);

	// Choose events to monitor: only those enabled
	int eventsToMonitor = 0;
	for (unsigned short i = 0; i < enabledEvents.size(); ++i) {
		short id = enabledEvents[i];
		if (id >= HWCounters::PQOS_MIN_EVENT && id <= HWCounters::PQOS_MAX_EVENT) {
			_enabledEvents[id - HWCounters::PQOS_MIN_EVENT] = true;

			if (std::string(HWCounters::counterDescriptions[id]) == "PQOS_PERF_EVENT_IPC") {
				eventsToMonitor |= PQOS_PERF_EVENT_IPC;
			} else if (std::string(HWCounters::counterDescriptions[id]) == "PQOS_PERF_EVENT_LLC_MISS") {
				eventsToMonitor |= PQOS_PERF_EVENT_LLC_MISS;
			} else if (std::string(HWCounters::counterDescriptions[id]) == "PQOS_MON_EVENT_LMEM_BW") {
				eventsToMonitor |= PQOS_MON_EVENT_LMEM_BW;
			} else if (std::string(HWCounters::counterDescriptions[id]) == "PQOS_MON_EVENT_RMEM_BW") {
				eventsToMonitor |= PQOS_MON_EVENT_RMEM_BW;
			} else if (std::string(HWCounters::counterDescriptions[id]) == "PQOS_MON_EVENT_L3_OCCUP") {
				eventsToMonitor |= PQOS_MON_EVENT_L3_OCCUP;
			} else {
				assert(false);
			}
		}
	}

	// Check which events are available in the system
	enum pqos_mon_event availableEvents = (pqos_mon_event) 0;
	for (size_t i = 0; i < pqosCapabilities->u.mon->num_events; ++i) {
		availableEvents = (pqos_mon_event) (availableEvents | (pqosCapabilities->u.mon->events[i].type));
	}

	// Only choose events that are enabled AND available
	_monitoredEvents = (pqos_mon_event) (availableEvents & eventsToMonitor);

	// If none of the events can be monitored, trigger an early shutdown
	_enabled = (_monitoredEvents != ((pqos_mon_event) 0));
}

PQoSHardwareCounters::~PQoSHardwareCounters()
{
	_enabled = false;

	// Shutdown PQoS monitoring
	int ret = pqos_fini();
	FatalErrorHandler::failIf(
		ret != PQOS_RETVAL_OK,
		ret, " when shutting down the PQoS library"
	);
}

void PQoSHardwareCounters::cpuBecomesIdle(
	CPUHardwareCountersInterface *cpuCounters,
	ThreadHardwareCountersInterface *threadCounters
) {
	if (_enabled) {
		// First read counters
		PQoSThreadHardwareCounters *pqosThreadCounters = (PQoSThreadHardwareCounters *) threadCounters;
		PQoSCPUHardwareCounters *pqosCPUCounters = (PQoSCPUHardwareCounters *) cpuCounters;
		assert(pqosThreadCounters != nullptr);
		assert(pqosCPUCounters != nullptr);

		// Poll PQoS events from the current thread only
		pqos_mon_data *threadData = pqosThreadCounters->getData();
		int ret = pqos_mon_poll(&threadData, 1);
		FatalErrorHandler::failIf(
			ret != PQOS_RETVAL_OK,
			ret, " when polling PQoS events for a task (start)"
		);

		// Copy read values for CPU counters
		pqosCPUCounters->readCounters(threadData);
	}
}

void PQoSHardwareCounters::threadInitialized(ThreadHardwareCountersInterface *threadCounters)
{
	if (_enabled) {
		PQoSThreadHardwareCounters *pqosCounters = (PQoSThreadHardwareCounters *) threadCounters;
		assert(pqosCounters != nullptr);

		// Allocate PQoS event structures
		pqos_mon_data *threadData = (pqos_mon_data *) malloc(sizeof(pqos_mon_data));
		FatalErrorHandler::failIf(
			threadData == nullptr,
			"Could not allocate memory for thread hardware counters"
		);

		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);

		// Link the structures to the current thread
		pqosCounters->setData(threadData);
		pqosCounters->setTid(currentThread->getTid());

		// Begin reading hardware counters for the thread
		int ret = pqos_mon_start_pid(
			pqosCounters->getTid(),
			_monitoredEvents,
			nullptr,
			pqosCounters->getData()
		);
		FatalErrorHandler::failIf(
			ret != PQOS_RETVAL_OK,
			ret, " when initializing hardware counters for a thread"
		);
	}
}

void PQoSHardwareCounters::threadShutdown(ThreadHardwareCountersInterface *threadCounters)
{
	if (_enabled) {
		PQoSThreadHardwareCounters *pqosCounters = (PQoSThreadHardwareCounters *) threadCounters;
		assert(pqosCounters != nullptr);

		// Finish PQoS monitoring for the current thread
		int ret = pqos_mon_stop(pqosCounters->getData());
		FatalErrorHandler::failIf(
			ret != PQOS_RETVAL_OK,
			ret, " when stopping hardware counters for a thread"
		);
	}
}

void PQoSHardwareCounters::taskReinitialized(TaskHardwareCountersInterface *taskCounters)
{
	if (_enabled) {
		PQoSTaskHardwareCounters *pqosCounters = (PQoSTaskHardwareCounters *) taskCounters;
		assert(pqosCounters != nullptr);

		pqosCounters->clear();
	}
}

void PQoSHardwareCounters::taskStarted(
	CPUHardwareCountersInterface *cpuCounters,
	ThreadHardwareCountersInterface *threadCounters,
	TaskHardwareCountersInterface *taskCounters
) {
	if (_enabled) {
		// First read counters
		PQoSThreadHardwareCounters *pqosThreadCounters = (PQoSThreadHardwareCounters *) threadCounters;
		PQoSTaskHardwareCounters *pqosTaskCounters = (PQoSTaskHardwareCounters *) taskCounters;
		PQoSCPUHardwareCounters *pqosCPUCounters = (PQoSCPUHardwareCounters *) cpuCounters;
		assert(pqosThreadCounters != nullptr);
		assert(pqosTaskCounters != nullptr);
		assert(pqosCPUCounters != nullptr);

		// Poll PQoS events from the current thread only
		pqos_mon_data *threadData = pqosThreadCounters->getData();
		int ret = pqos_mon_poll(&threadData, 1);
		FatalErrorHandler::failIf(
			ret != PQOS_RETVAL_OK,
			ret, " when polling PQoS events for a task (start)"
		);

		// Copy read values for CPU counters
		pqosCPUCounters->readCounters(threadData);

		// Copy read values for Task counters
		if (pqosTaskCounters->isEnabled()) {
			if (!pqosTaskCounters->isActive()) {
				pqosTaskCounters->startReading(threadData);
			}
		}
	}
}

void PQoSHardwareCounters::taskStopped(
	CPUHardwareCountersInterface *cpuCounters,
	ThreadHardwareCountersInterface *threadCounters,
	TaskHardwareCountersInterface *taskCounters
) {
	if (_enabled) {
		// First read counters
		PQoSThreadHardwareCounters *pqosThreadCounters = (PQoSThreadHardwareCounters *) threadCounters;
		PQoSTaskHardwareCounters *pqosTaskCounters = (PQoSTaskHardwareCounters *) taskCounters;
		PQoSCPUHardwareCounters *pqosCPUCounters = (PQoSCPUHardwareCounters *) cpuCounters;
		assert(pqosThreadCounters != nullptr);
		assert(pqosTaskCounters != nullptr);
		assert(pqosCPUCounters != nullptr);

		pqos_mon_data *threadData = pqosThreadCounters->getData();

		// Poll PQoS events from the current thread only
		int ret = pqos_mon_poll(&threadData, 1);
		FatalErrorHandler::failIf(
			ret != PQOS_RETVAL_OK,
			ret, " when polling PQoS events for a task (stop)"
		);

		// Copy read values for CPU counters
		pqosCPUCounters->readCounters(threadData);

		// Copy read values for Task counters
		if (pqosTaskCounters->isEnabled()) {
			if (pqosTaskCounters->isActive()) {
				pqosTaskCounters->stopReading(threadData);
			}
		}
	}
}

