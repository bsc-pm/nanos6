/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef THROTTLE_HPP
#define THROTTLE_HPP

#include "lowlevel/EnvironmentVariable.hpp"

class Task;
class WorkerThread;

class Throttle {
private:
	static int _pressure;
	static EnvironmentVariable<bool> _enabled;
	static EnvironmentVariable<int> _throttleTasks;
	static EnvironmentVariable<int> _throttlePressure;
	static EnvironmentVariable<StringifiedMemorySize> _throttleMem;

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
	//!
	//! \returns 0
	static int evaluate(void *);

	//! \brief Initializes the Throttle status and registers polling services.
	//!
	//! Should be called before any other throttle function
	static void initialize();

	//! \brief Shuts down the throttle and frees any additional resources
	//!
	//! No other throttle functions can be called after shutdown()
	static void shutdown();

	//! \brief Engage if the conditions of the task require it the throttle mechanism
	//!
	//! \returns true if the throttle should be engaged again, false if the task can continue
	static bool engage(Task *task, WorkerThread *workerThread);
};

#endif // THROTTLE_HPP
