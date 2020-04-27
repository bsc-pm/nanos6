/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <sys/utsname.h>

#include "PQoSHardwareCounters.hpp"
#include "PQoSThreadHardwareCounters.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware-counters/TaskHardwareCounters.hpp"
#include "hardware-counters/ThreadHardwareCounters.hpp"


PQoSHardwareCounters::PQoSHardwareCounters(bool verbose, const std::string &verboseFile)
{
	_verbose = verbose;
	_verboseFile = verboseFile;

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

	// Display statistics
	if (_verbose) {
		displayStatistics();
	}
}

void PQoSHardwareCounters::threadInitialized()
{
	if (_enabled) {
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);

		ThreadHardwareCounters *threadCounters = thread->getHardwareCounters();
		assert(threadCounters != nullptr);

		PQoSThreadHardwareCounters *pqosCounters =
			(PQoSThreadHardwareCounters *) threadCounters->getPQoSCounters();
		assert(pqosCounters != nullptr);

		// Allocate PQoS event structures
		pqos_mon_data *threadData = (pqos_mon_data *) malloc(sizeof(pqos_mon_data));
		FatalErrorHandler::failIf(
			threadData == nullptr,
			"Could not allocate memory for thread hardware counters"
		);

		// Link the structures to the current thread
		pqosCounters->setData(threadData);
		pqosCounters->setTid(thread->getTid());

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

void PQoSHardwareCounters::threadShutdown()
{
	if (_enabled) {
		WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
		assert(thread != nullptr);

		ThreadHardwareCounters *threadCounters = thread->getHardwareCounters();
		assert(threadCounters != nullptr);

		PQoSThreadHardwareCounters *pqosCounters =
			(PQoSThreadHardwareCounters *) threadCounters->getPQoSCounters();
		assert(pqosCounters != nullptr);

		// Finish PQoS monitoring for the current thread
		int ret = pqos_mon_stop(pqosCounters->getData());
		FatalErrorHandler::failIf(
			ret != PQOS_RETVAL_OK,
			ret, " when stopping hardware counters for a thread"
		);

		delete pqosCounters;
	}
}

void PQoSHardwareCounters::taskCreated(Task *task, bool enabled)
{
	if (_enabled) {
		assert(task != nullptr);

		if (enabled) {
			if (_verbose) {
				std::string tasktype = task->getLabel();

				_statsLock.lock();
				statistics_map_t::iterator it = _statistics.find(tasktype);
				if (it == _statistics.end()) {
					_statistics.emplace(
						std::piecewise_construct,
						std::forward_as_tuple(tasktype),
						std::forward_as_tuple(HWCounters::PQOS_NUM_EVENTS)
					);
				}
				_statsLock.unlock();
			}
		}
	}
}

void PQoSHardwareCounters::taskReinitialized(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);

		TaskHardwareCounters *taskCounters = task->getHardwareCounters();
		assert(taskCounters != nullptr);

		PQoSTaskHardwareCounters *pqosCounters =
			(PQoSTaskHardwareCounters *) taskCounters->getPQoSCounters();
		assert(pqosCounters != nullptr);

		pqosCounters->clear();
	}
}

void PQoSHardwareCounters::taskStarted(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);

		TaskHardwareCounters *taskCounters = task->getHardwareCounters();
		assert(taskCounters != nullptr);

		PQoSTaskHardwareCounters *pqosTaskCounters =
			(PQoSTaskHardwareCounters *) taskCounters->getPQoSCounters();
		assert(pqosTaskCounters != nullptr);

		if (pqosTaskCounters->isEnabled()) {
			if (!pqosTaskCounters->isActive()) {
				WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
				assert(thread != nullptr);

				ThreadHardwareCounters *threadCounters = thread->getHardwareCounters();
				assert(threadCounters != nullptr);

				PQoSThreadHardwareCounters *pqosThreadCounters =
					(PQoSThreadHardwareCounters *) threadCounters->getPQoSCounters();
				assert(pqosThreadCounters != nullptr);

				// Poll PQoS events from the current thread only
				pqos_mon_data *threadData = pqosThreadCounters->getData();
				int ret = pqos_mon_poll(&threadData, 1);
				FatalErrorHandler::failIf(
					ret != PQOS_RETVAL_OK,
					ret, " when polling PQoS events for a task (start)"
				);

				// If successfull, save counters when the task starts or resumes execution
				pqosTaskCounters->startReading(threadData);
			}
		}
	}
}

void PQoSHardwareCounters::taskStopped(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);

		TaskHardwareCounters *taskCounters = task->getHardwareCounters();
		assert(taskCounters != nullptr);

		PQoSTaskHardwareCounters *pqosTaskCounters =
			(PQoSTaskHardwareCounters *) taskCounters->getPQoSCounters();
		assert(pqosTaskCounters != nullptr);

		if (pqosTaskCounters->isEnabled()) {
			if (pqosTaskCounters->isActive()) {
				WorkerThread *thread = WorkerThread::getCurrentWorkerThread();
				assert(thread != nullptr);

				ThreadHardwareCounters *threadCounters = thread->getHardwareCounters();
				assert(threadCounters != nullptr);

				PQoSThreadHardwareCounters *pqosThreadCounters =
					(PQoSThreadHardwareCounters *) threadCounters->getPQoSCounters();
				assert(pqosThreadCounters != nullptr);

				pqos_mon_data *threadData = pqosThreadCounters->getData();

				// Poll PQoS events from the current thread only
				int ret = pqos_mon_poll(&threadData, 1);
				FatalErrorHandler::failIf(
					ret != PQOS_RETVAL_OK,
					ret, " when polling PQoS events for a task (stop)"
				);

				pqosTaskCounters->stopReading(threadData);
			}
		}
	}
}

void PQoSHardwareCounters::taskFinished(Task *task)
{
	if (_enabled) {
		assert(task != nullptr);

		TaskHardwareCounters *taskCounters = task->getHardwareCounters();
		assert(taskCounters != nullptr);

		PQoSTaskHardwareCounters *pqosTaskCounters =
			(PQoSTaskHardwareCounters *) taskCounters->getPQoSCounters();
		assert(pqosTaskCounters != nullptr);

		if (pqosTaskCounters->isEnabled()) {
			if (_verbose) {
				std::string tasktype = task->getLabel();

				_statsLock.lock();
				statistics_map_t::iterator it = _statistics.find(tasktype);
				assert(it != _statistics.end());

				it->second[HWCounters::PQOS_MON_EVENT_L3_OCCUP](
					pqosTaskCounters->getAccumulated(HWCounters::PQOS_MON_EVENT_L3_OCCUP)
				);
				it->second[HWCounters::PQOS_PERF_EVENT_IPC](
					pqosTaskCounters->getAccumulated(HWCounters::PQOS_PERF_EVENT_IPC)
				);
				it->second[HWCounters::PQOS_MON_EVENT_LMEM_BW](
					pqosTaskCounters->getAccumulated(HWCounters::PQOS_MON_EVENT_LMEM_BW)
				);
				it->second[HWCounters::PQOS_MON_EVENT_RMEM_BW](
					pqosTaskCounters->getAccumulated(HWCounters::PQOS_MON_EVENT_RMEM_BW)
				);
				it->second[HWCounters::PQOS_PERF_EVENT_LLC_MISS](
					pqosTaskCounters->getAccumulated(HWCounters::PQOS_PERF_EVENT_LLC_MISS)
				);

				_statsLock.unlock();
			}
		}
	}
}

void PQoSHardwareCounters::displayStatistics()
{
	// Try opening the output file
	std::ios_base::openmode openMode = std::ios::out;
	std::ofstream output(_verboseFile, openMode);
	FatalErrorHandler::warnIf(
		!output.is_open(),
		"Could not create or open the verbose file: ", _verboseFile, ". ",
		"Using standard output."
	);

	// Retrieve statistics
	std::stringstream outputStream;
	outputStream << std::left << std::fixed << std::setprecision(5);
	outputStream << "-------------------------------\n";

	// Iterate through all tasktypes
	for (auto &it : _statistics) {
		size_t instances = BoostAcc::count(it.second[0]);
		if (instances > 0) {
			std::string typeLabel = it.first + " (" + std::to_string(instances) + ")";

			outputStream <<
				std::setw(7)  << "STATS"                 << " " <<
				std::setw(6)  << "PQOS"                  << " " <<
				std::setw(39) << "TASK-TYPE (INSTANCES)" << " " <<
				std::setw(30) << typeLabel               << "\n";

			// Iterate through all counter types
			for (unsigned short id = HWCounters::PQOS_MIN_EVENT; id <= HWCounters::PQOS_MAX_EVENT; ++id) {
				double counterAvg   = BoostAcc::mean(it.second[id]);
				double counterStdev = sqrt(BoostAcc::variance(it.second[id]));
				double counterSum   = BoostAcc::sum(it.second[id]);

				// In KB
				if (id == HWCounters::PQOS_MON_EVENT_L3_OCCUP ||
					id == HWCounters::PQOS_MON_EVENT_LMEM_BW  ||
					id == HWCounters::PQOS_MON_EVENT_RMEM_BW
				) {
					counterAvg   /= 1024.0;
					counterStdev /= 1024.0;
					counterSum   /= 1024.0;
				}

				outputStream <<
					std::setw(7)  << "STATS"                             << " "   <<
					std::setw(6)  << "PQOS"                              << " "   <<
					std::setw(39) << HWCounters::counterDescriptions[id] << " "   <<
					std::setw(30) << "SUM / AVG / STDEV"                 << " "   <<
					std::setw(15) << counterSum                          << " / " <<
					std::setw(15) << counterAvg                          << " / " <<
					std::setw(15) << counterStdev                        << "\n";
			}

			outputStream << "-------------------------------\n";
		}
	}

	if (output.is_open()) {
		// Output into the file and close it
		output << outputStream.str();
		output.close();
	} else {
		std::cout << outputStream.str();
	}
}
