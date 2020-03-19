/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <sys/utsname.h>

#include "PQoSHardwareCountersImplementation.hpp"
#include "PQoSTaskHardwareCounters.hpp"
#include "PQoSThreadHardwareCounters.hpp"
#include "executors/threads/WorkerThread.hpp"


PQoSHardwareCountersImplementation::statistics_map_t PQoSHardwareCountersImplementation::_statistics;
SpinLock PQoSHardwareCountersImplementation::_statsLock;
bool PQoSHardwareCountersImplementation::_enabled;
bool PQoSHardwareCountersImplementation::_verbose;
std::string PQoSHardwareCountersImplementation::_verboseFile;
enum pqos_mon_event _monitoredEvents;


void PQoSHardwareCountersImplementation::initialize(bool verbose, std::string verboseFile)
{
	_verbose = verbose;
	_verboseFile = verboseFile;

	// Check if the PQoS version may give problems
	utsname kernelInfo;
	if (uname(&kernelInfo) == 0) {
		std::string kernelRelease(kernelInfo.release);
		if (kernelRelease.find("4.13.", 0) == 0) {
			if (kernelRelease.find("4.13.0", 0) != 0) {
				FatalErrorHandler::warnIf(
					true,
					"4.13.X (X != 0) kernel versions may give incorrect readings for MBM counters"
				);
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
		"Error '", ret, "' when initializing the PQoS library"
	);
	ret = pqos_cap_get(&pqosCap, &pqosCPUInfo);
	FatalErrorHandler::failIf(
		ret != PQOS_RETVAL_OK,
		"Error '", ret, "' when retrieving PQoS capabilities"
	);
	ret = pqos_cap_get_type(pqosCap, PQOS_CAP_TYPE_MON, &pqosCapabilities);
	FatalErrorHandler::failIf(
		ret != PQOS_RETVAL_OK,
		"Error '", ret, "' when retrieving PQoS capability types"
	);

	assert(pqosCapabilities != nullptr);
	assert(pqosCapabilities->u.mon != nullptr);

	// Choose events to monitor
	const enum pqos_mon_event eventsToMonitor = (pqos_mon_event)
	(
		PQOS_PERF_EVENT_IPC      | // IPC
		PQOS_PERF_EVENT_LLC_MISS | // LLC Misses
		PQOS_MON_EVENT_LMEM_BW   | // Local Memory Bandwidth
		PQOS_MON_EVENT_RMEM_BW   | // Remote Memory Bandwidth
		PQOS_MON_EVENT_L3_OCCUP    // LLC Usage
	);

	enum pqos_mon_event availableEvents = (pqos_mon_event) 0;
	for (size_t i = 0; i < pqosCapabilities->u.mon->num_events; ++i) {
		availableEvents = (pqos_mon_event) (availableEvents | (pqosCapabilities->u.mon->events[i].type));
	}

	// Only choose events we want to monitor that are available
	_monitoredEvents = (pqos_mon_event) (availableEvents & eventsToMonitor);

	// If none of the events can be monitored, trigger an early shutdown
	if (_monitoredEvents == ((pqos_mon_event) 0)) {
		shutdown();
	} else {
		_enabled = true;
	}
}

void PQoSHardwareCountersImplementation::shutdown()
{
	_enabled = false;

	// Shutdown PQoS monitoring
	int ret = pqos_fini();
	FatalErrorHandler::failIf(
		ret != PQOS_RETVAL_OK,
		"Error '", ret, "' when shutting down the PQoS library"
	);

	// Display statistics
	if (_verbose) {
		displayStatistics();
	}
}

void PQoSHardwareCountersImplementation::threadInitialized()
{
	if (_enabled) {
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);

		PQoSThreadHardwareCounters *threadCounters = new PQoSThreadHardwareCounters();
		thread->setHardwareCounters(threadCounters);
		assert(threadCounters != nullptr);

		// Allocate PQoS event structures
		pqos_mon_data *threadData = (pqos_mon_data *) malloc(sizeof(pqos_mon_data));
		FatalErrorHandler::failIf(
			threadData == nullptr,
			"Could not allocate memory for thread hardware counters"
		);

		// Link the structures to the current thread
		threadCounters->setData(threadData);
		threadCounters->setTid(thread->getTid());

		// Begin reading hardware counters for the thread
		int ret = pqos_mon_start_pid(
			threadCounters->getTid(),
			_monitoredEvents,
			nullptr,
			threadCounters->getData()
		);
		FatalErrorHandler::failIf(
			ret != PQOS_RETVAL_OK,
			"Error '", ret, "' when initializing hardware counters for a thread"
		);
	}
}

void PQoSHardwareCountersImplementation::threadShutdown()
{
	if (_enabled) {
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);

		PQoSThreadHardwareCounters *threadCounters = (PQoSThreadHardwareCounters *) thread->getHardwareCounters();
		assert(threadCounters != nullptr);

		// Finish PQoS monitoring for the current thread
		int ret = pqos_mon_stop(threadCounters->getData());
		FatalErrorHandler::failIf(
			ret != PQOS_RETVAL_OK,
			"Error '", ret, "' when stopping hardware counters for a thread"
		);

		delete threadCounters;
	}
}

void PQoSHardwareCountersImplementation::taskCreated(Task *task, bool enabled)
{
	if (_enabled) {
		assert(task != nullptr);

		// TODO/FIXME: At task instantiation time!
		PQoSTaskHardwareCounters *taskCounters = new PQoSTaskHardwareCounters();
		task->setHardwareCounters(taskCounters);
		assert(taskCounters != nullptr);

		taskCounters->setEnabled(enabled);

		if (enabled) {
			if (_verbose) {
				std::string tasktype = task->getLabel();

				_statsLock.lock();
				statistics_map_t::iterator it = _statistics.find(tasktype);
				if (it == _statistics.end()) {
					_statistics.emplace(
						std::piecewise_construct,
						std::forward_as_tuple(tasktype),
						std::forward_as_tuple(num_pqos_counters)
					);
				}
				_statsLock.unlock();
			}
		}
	}
}

void PQoSHardwareCountersImplementation::taskStarted(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);

		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		if (thread != nullptr) {
			PQoSTaskHardwareCounters *taskCounters = (PQoSTaskHardwareCounters *) task->getHardwareCounters();
			assert(taskCounters != nullptr);

			if (taskCounters->isEnabled()) {
				if (!taskCounters->isActive()) {
					PQoSThreadHardwareCounters *threadCounters = (PQoSThreadHardwareCounters *) thread->getHardwareCounters();
					assert(threadCounters != nullptr);

					// Poll PQoS events from the current thread only
					pqos_mon_data *threadData = threadCounters->getData();
					int ret = pqos_mon_poll(&threadData, 1);
					FatalErrorHandler::failIf(
						ret != PQOS_RETVAL_OK,
						"Error '", ret, "' when polling PQoS events for a task (start)"
					);

					// If successfull, save counters when the task starts or resumes execution
					taskCounters->startReading(threadData);
				}
			}
		}
	}
}

void PQoSHardwareCountersImplementation::taskStopped(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);

		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		if (thread != nullptr) {
			PQoSTaskHardwareCounters *taskCounters = (PQoSTaskHardwareCounters *) task->getHardwareCounters();
			assert(taskCounters != nullptr);

			if (taskCounters->isEnabled()) {
				if (taskCounters->isActive()) {
					PQoSThreadHardwareCounters *threadCounters = (PQoSThreadHardwareCounters *) thread->getHardwareCounters();
					assert(threadCounters != nullptr);

					pqos_mon_data *threadData = threadCounters->getData();

					// Poll PQoS events from the current thread only
					int ret = pqos_mon_poll(&threadData, 1);
					FatalErrorHandler::failIf(
						ret != PQOS_RETVAL_OK,
						"Error '", ret, "' when polling PQoS events for a task (stop)"
					);

					taskCounters->stopReading(threadData);
				}
			}
		}
	}
}

void PQoSHardwareCountersImplementation::taskFinished(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);

		// TODO/FIXME: At task instantiation time, no need to destroy, do it at task destructor!
		PQoSTaskHardwareCounters *taskCounters = (PQoSTaskHardwareCounters *) task->getHardwareCounters();
		assert(taskCounters != nullptr);

		if (taskCounters->isEnabled()) {
			if (_verbose) {
				std::string tasktype = task->getLabel();

				_statsLock.lock();
				statistics_map_t::iterator it = _statistics.find(tasktype);
				assert(it != _statistics.end());

				it->second[pqos_llc_usage           ](taskCounters->getAccumulated(HWCounters::llc_usage));
				it->second[pqos_ipc                 ](taskCounters->getAccumulated(HWCounters::ipc));
				it->second[pqos_local_mem_bandwidth ](taskCounters->getAccumulated(HWCounters::local_mem_bandwidth));
				it->second[pqos_remote_mem_bandwidth](taskCounters->getAccumulated(HWCounters::remote_mem_bandwidth));
				it->second[pqos_llc_miss_rate       ](taskCounters->getAccumulated(HWCounters::llc_miss_rate));
				_statsLock.unlock();
			}
		}

		delete taskCounters;
	}
}
