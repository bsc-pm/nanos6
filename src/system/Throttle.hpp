/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef THROTTLE_HPP
#define THROTTLE_HPP

#include <atomic>

#include "support/config/ConfigVariable.hpp"

class Task;
class WorkerThread;

class Throttle {
private:
	static int _pressure;
	static ConfigVariable<bool> _enabled;
	static ConfigVariable<int> _throttleTasks;
	static ConfigVariable<int> _throttlePressure;
	static ConfigVariable<StringifiedMemorySize> _throttleMem;
	static ConfigVariable<size_t> _throttlePollingPeriod;

	static std::atomic<bool> _stopService;
	static std::atomic<bool> _finishedService;

	static int getAllowedTasks(int nestingLevel);

public:
	//! \brief Checks if the throttle is in active mode and should be engaged
	//!
	//! \returns true if the throttle should be engaged, false otherwise
	static inline bool isActive()
	{
		return _enabled;
	}

	//! \brief Evaluates current system status and sets throttle activation
	static void evaluate(void *);

	//! \brief Executed when the service is completed
	static void complete(void *);

	//! \brief Initializes the Throttle status and registers a polling task
	//!
	//! Should be called before any other throttle function
	static void initialize();

	//! \brief Shuts down the throttle and frees any additional resources
	//!
	//! No other throttle functions can be called after shutdown()
	static void shutdown();

	//! \brief Engage if the conditions of the creator task require the throttle mechanism
	//!
	//! \param creator The task that is creating a child task
	//! \param workerThread The worker thread executing the creator task
	//!
	//! \returns true if the throttle should be engaged again, false if the creator can continue
	static bool engage(Task *creator, WorkerThread *workerThread);
};

#endif // THROTTLE_HPP
