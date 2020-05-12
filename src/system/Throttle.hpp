/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

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
	// Determines if throttle is in active mode
	static inline bool active() {
		return _enabled;
	}

	// Evaluates current system status and sets throttle activation
	static int evaluate(void *);

	// Initialize the throttle
	static void initialize();

	// Unregister the throttle
	static void shutdown();

	static bool engage(Task *task, WorkerThread *workerThread);
};