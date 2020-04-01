/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/task-info-registration.h>

#include "ompss/TaskInfo.hpp"


extern "C" void nanos6_register_task_info(nanos6_task_info_t *task_info)
{
	TaskInfo::registerTaskInfo(task_info);
}
