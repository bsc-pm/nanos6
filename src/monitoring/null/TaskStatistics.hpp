/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef NULL_TASK_STATISTICS_HPP
#define NULL_TASK_STATISTICS_HPP

enum monitoring_task_status_t {
	pending_status = 0,
	ready_status,
	executing_status,
	blocked_status,
	runtime_status,
	num_status,
	null_status = -1
};


class TaskStatistics {
};

#endif // NULL_TASK_STATISTICS_HPP
