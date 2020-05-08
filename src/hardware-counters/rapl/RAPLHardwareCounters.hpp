/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef RAPL_HARDWARE_COUNTERS_HPP
#define RAPL_HARDWARE_COUNTERS_HPP

#include <cstdio>

#include "hardware-counters/HardwareCountersInterface.hpp"

#define RAPL_NUM_DOMAINS 5
#define RAPL_MAX_PACKAGES 16
#define RAPL_BUFFER_SIZE 256


class CPUHardwareCountersInterface;
class Task;
class TaskHardwareCountersInterface;
class ThreadHardwareCountersInterface;

class RAPLHardwareCounters : public HardwareCountersInterface {

private:

	//! Whether the verbose mode is activated
	bool _verbose;

	//! The file name on which to output statistics when verbose is enabled
	std::string _verboseFile;

	//! The number of CPUs in the system
	size_t _numCPUs;

	//! The number of packages in the system
	size_t _numPackages;

	//! Event names for all packages and domains
	char _eventNames[RAPL_MAX_PACKAGES][RAPL_NUM_DOMAINS][RAPL_BUFFER_SIZE];

	//! File names for all packages and domains
	char _fileNames[RAPL_MAX_PACKAGES][RAPL_NUM_DOMAINS][RAPL_BUFFER_SIZE];

	//! Whether a certain event is valid
	bool _validEvents[RAPL_MAX_PACKAGES][RAPL_NUM_DOMAINS];

	//! Values read at initialization time (start of the execution)
	size_t _startValues[RAPL_MAX_PACKAGES][RAPL_NUM_DOMAINS];

	//! Values read at shutdown time (end of the execution)
	size_t _finishValues[RAPL_MAX_PACKAGES][RAPL_NUM_DOMAINS];

private:

	//! \brief Detect if the current CPU architecture is compatible with RAPL
	void raplDetectCPU();

	//! \brief Detect the number of cores and CPU sockets
	void raplDetectPackages();

	//! \brief Initialize and start the power library
	void raplInitialize();

	//! \brief Read power counters
	//!
	//! \param[out] values The arrays where power counters will be written to
	void raplReadCounters(size_t values[RAPL_MAX_PACKAGES][RAPL_NUM_DOMAINS]);

	//! \brief Shutdown the power library
	void raplShutdown();

	//! \brief Print power usage information
	void displayStatistics() const;

public:

	RAPLHardwareCounters(bool verbose, const std::string &verboseFile);

	~RAPLHardwareCounters();

	inline void cpuBecomesIdle(CPUHardwareCountersInterface *, ThreadHardwareCountersInterface *)
	{
	}

	inline void threadInitialized(ThreadHardwareCountersInterface *)
	{
	}

	inline void threadShutdown(ThreadHardwareCountersInterface *)
	{
	}

	inline void taskCreated(Task *, bool)
	{
	}

	inline void taskReinitialized(TaskHardwareCountersInterface *)
	{
	}

	inline void taskStarted(
		CPUHardwareCountersInterface *,
		ThreadHardwareCountersInterface *,
		TaskHardwareCountersInterface *
	) {
	}

	inline void taskStopped(
		CPUHardwareCountersInterface *,
		ThreadHardwareCountersInterface *,
		TaskHardwareCountersInterface *
	) {
	}

	inline void taskFinished(Task *, TaskHardwareCountersInterface *)
	{
	}

};

#endif // RAPL_HARDWARE_COUNTERS_HPP
