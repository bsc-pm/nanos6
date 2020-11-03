/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "TrackingPoints.hpp"
#include "tasks/Taskfor.hpp"

void TrackingPoints::taskIsExecuting(Task *task)
{
	HardwareCounters::updateRuntimeCounters();

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	if (task->isTaskforCollaborator()) {
		bool first = ((Taskfor *) task)->hasFirstChunk();
		Task *parent = task->getParent();
		assert(parent != nullptr);

		Instrument::task_id_t parentId = parent->getInstrumentationTaskId();
		Instrument::startTaskforCollaborator(parentId, taskId, first);
		Instrument::taskforCollaboratorIsExecuting(parentId, taskId);
	} else {
		Instrument::startTask(taskId);
		Instrument::taskIsExecuting(taskId);
	}

	Monitoring::taskChangedStatus(task, executing_status);
}

void TrackingPoints::taskCompletedUserCode(Task *task)
{
	assert(task != nullptr);

	if (task->hasCode()) {
		HardwareCounters::updateTaskCounters(task);
		Monitoring::taskChangedStatus(task, paused_status);
		Monitoring::taskCompletedUserCode(task);

		Instrument::task_id_t taskId = task->getInstrumentationTaskId();
		if (task->isTaskforCollaborator()) {
			bool last = ((Taskfor *) task)->hasLastChunk();
			Task *parent = task->getParent();
			assert(parent != nullptr);

			Instrument::task_id_t parentTaskId = parent->getInstrumentationTaskId();
			Instrument::taskforCollaboratorStopped(parentTaskId, taskId);
			Instrument::endTaskforCollaborator(parentTaskId, taskId, last);
		} else {
			Instrument::taskIsZombie(taskId);
			Instrument::endTask(taskId);
		}
	} else {
		Monitoring::taskChangedStatus(task, paused_status);
		Monitoring::taskCompletedUserCode(task);
	}
}
