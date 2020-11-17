/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef MONITORING_SUPPORT_HPP
#define MONITORING_SUPPORT_HPP


#define DEFAULT_COST 1

#define PREDICTION_UNAVAILABLE -1.0


enum monitoring_task_status_t {
	// The task is ready to be executed
	ready_status = 0,
	// The task is being executed
	executing_status,
	// An aggregation of runtime + pending + blocked
	paused_status,
	num_status,
	null_status = -1
};


#endif // MONITORING_SUPPORT_HPP
