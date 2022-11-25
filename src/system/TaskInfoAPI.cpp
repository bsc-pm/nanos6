/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/task-info-registration.h>

#include "monitoring/Monitoring.hpp"


extern "C" void nanos6_register_task_info(nanos6_task_info_t *task_info)
{
	Monitoring::registerTasktype(task_info);
}
