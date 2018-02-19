/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKLOOP_SCHEDULING_POLICY_HPP
#define TASKLOOP_SCHEDULING_POLICY_HPP

#include "lowlevel/EnvironmentVariable.hpp"

class TaskloopSchedulingPolicy {
public:
	static inline bool isRequeueEnabled()
	{
		static EnvironmentVariable<bool> _requeueTaskloop("NANOS6_REQUEUE_TASKLOOP", false);
		return _requeueTaskloop.getValue();
	}
};

#endif // TASKLOOP_SCHEDULING_POLCIY_HPP
