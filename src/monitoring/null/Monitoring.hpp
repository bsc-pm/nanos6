/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef NULL_MONITORING_HPP
#define NULL_MONITORING_HPP

#include "tasks/Task.hpp"


class Monitoring {

public:
	
	static inline void initialize()
	{
	}
	
	static inline void shutdown()
	{
	}
	
	static inline bool isEnabled()
	{
		return false;
	}
	
	static inline void taskCreated(Task *)
	{
	}
	
	static inline void taskChangedStatus(Task *, monitoring_task_status_t, ComputePlace * = nullptr)
	{
	}
	
	static inline void taskCompletedUserCode(Task *, ComputePlace *)
	{
	}
	
	static inline void taskFinished(Task *)
	{
	}
	
	static inline void initializeThread()
	{
	}
	
	static inline void shutdownThread()
	{
	}
	
	static inline double getPredictedElapsedTime()
	{
		return 0.0;
	}
	
};

#endif // NULL_MONITORING_HPP
