/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "TaskInfo.hpp"
#include "tasks/TaskTypeData.hpp"


#define MAX_TASKINFOS 500

std::vector<nanos6_task_info_t *> TaskInfo::_taskInfos(MAX_TASKINFOS, nullptr);
std::atomic<short> TaskInfo::_nextId(0);


void TaskInfo::registerTaskInfo(nanos6_task_info_t *taskInfo)
{
	assert(taskInfo != nullptr);

	// First check if this taskinfo is a duplicate
	nanos6_task_info_t *duplicateTaskInfo = nullptr;
	bool duplicated = isDuplicated(taskInfo, duplicateTaskInfo);

	short currentId = _nextId++;
	assert(currentId < MAX_TASKINFOS);

	_taskInfos[currentId] = taskInfo;

	if (duplicated) {
		assert(duplicateTaskInfo != nullptr);

		// If it's a duplicate, copy the data pointer
		taskInfo->task_type_data = duplicateTaskInfo->task_type_data;
	} else {
		// This is a new type of task, get a new id and create strucutres
		TaskTypeData *taskTypeData = new TaskTypeData(currentId);
		assert(taskTypeData != nullptr);

		taskInfo->task_type_data = taskTypeData;
	}
}

