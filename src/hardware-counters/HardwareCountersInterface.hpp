/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef HARDWARE_COUNTERS_INTERFACE_HPP
#define HARDWARE_COUNTERS_INTERFACE_HPP


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

	//! \brief Initialize hardware counter structures for a task
	//!
	//! \param[out] task The newly created task
	//! \param[in] enabled Whether to create structures and monitor this task
	virtual void taskCreated(Task *task, bool enabled) = 0;

	//! \brief Reinitialize all hardware counter structures for a task
	//!
	//! \param[out] taskCounters The hardware counter structure to reinitialize
	virtual void taskReinitialized(TaskHardwareCountersInterface *taskCounters) = 0;

	//! \brief Start reading hardware counters for a task
	//!
	//! \param[out] threadCounters The hardware counter structures of the thread executing the task
	//! \param[out] taskCounters The hardware counter structure of the task to start
	virtual void taskStarted(
		ThreadHardwareCountersInterface *threadCounters,
		TaskHardwareCountersInterface *taskCounters
	) = 0;

	//! \brief Stop reading hardware counters for a task
	//!
	//! \param[out] threadCounters The hardware counter structures of the thread executing the task
	//! \param[out] taskCounters The hardware counter structure of the task to stop
	virtual void taskStopped(
		ThreadHardwareCountersInterface *threadCounters,
		TaskHardwareCountersInterface *taskCounters
	) = 0;

	//! \brief Finish monitoring a task's hardware counters and accumulate them
	//!
	//! \param[out] task The task to finish hardware counters monitoring for
	//! \param[out] taskCounters The hardware counter structure of the task
	virtual void taskFinished(Task *task, TaskHardwareCountersInterface *taskCounters) = 0;

};

#endif // HARDWARE_COUNTERS_INTERFACE_HPP
