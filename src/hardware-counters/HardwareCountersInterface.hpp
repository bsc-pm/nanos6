/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef HARDWARE_COUNTERS_INTERFACE_HPP
#define HARDWARE_COUNTERS_INTERFACE_HPP


class CPUHardwareCountersInterface;
class Task;
class TaskHardwareCountersInterface;
class ThreadHardwareCountersInterface;

class HardwareCountersInterface {

public:

	virtual ~HardwareCountersInterface()
	{
	}

	//! \brief Initialize hardware counter structures for a new thread
	//!
	//! \param[out] threadCounters The hardware counter structures to initialize
	virtual void threadInitialized(ThreadHardwareCountersInterface *threadCounters) = 0;

	//! \brief Destroy the hardware counter structures of a thread
	//!
	//! \param[out] threadCounters The hardware counter structures to initialize
	virtual void threadShutdown(ThreadHardwareCountersInterface *threadCounters) = 0;

	//! \brief Update and read hardware counters for a task
	//!
	//! \param[out] threadCounters The hardware counter structures of the thread executing the task
	//! \param[out] taskCounters The hardware counter structure of the task to start
	virtual void updateTaskCounters(
		ThreadHardwareCountersInterface *threadCounters,
		TaskHardwareCountersInterface *taskCounters
	) = 0;

	//! \brief Update and read hardware counters for the runtime (current CPU)
	//!
	//! \param[out] cpuCounters The hardware counter structures of the CPU
	//! \param[out] threadCounters The hardware counter structures of the thread
	virtual void updateRuntimeCounters(
		CPUHardwareCountersInterface *cpuCounters,
		ThreadHardwareCountersInterface *threadCounters
	) = 0;

	//! \brief An optional function that displays statistics of the backend
	virtual void displayStatistics() const
	{
	}

};

#endif // HARDWARE_COUNTERS_INTERFACE_HPP
