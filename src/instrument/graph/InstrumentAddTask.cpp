/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/


#include <cassert>
#include <mutex>

#include "ExecutionSteps.hpp"
#include "InstrumentAddTask.hpp"
#include "InstrumentGraph.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <InstrumentInstrumentationContext.hpp>


namespace Instrument {
	using namespace Graph;
	
	
	task_id_t enterAddTask(
		__attribute__((unused)) nanos6_task_info_t *taskInfo,
		__attribute__((unused)) nanos6_task_invocation_info_t *taskInvokationInfo,
		__attribute__((unused)) size_t flags,
		InstrumentationContext const &context
	) {
		std::lock_guard<SpinLock> guard(_graphLock);
		
		// Get an ID for the task
		task_id_t taskId = _nextTaskId++;
		
		// Set up the parent phase
		if (context._taskId != task_id_t()) {
			task_info_t &parentInfo = _taskToInfoMap[context._taskId];
			
			parentInfo._hasChildren = true;
			
			task_group_t *taskGroup = nullptr;
			if (parentInfo._phaseList.empty()) {
				taskGroup = new task_group_t(_nextTaskwaitId++);
				parentInfo._phaseList.push_back(taskGroup);
			} else {
				phase_t *currentPhase = parentInfo._phaseList.back();
				
				taskGroup = dynamic_cast<task_group_t *> (currentPhase);
				if (taskGroup == nullptr) {
					// First task after a taskwait
					taskGroup = new task_group_t(_nextTaskwaitId++);
					parentInfo._phaseList.push_back(taskGroup);
				}
			}
		}
		
		create_task_step_t *createTaskStep = new create_task_step_t(context, taskId);
		_executionSequence.push_back(createTaskStep);
		
		return taskId;
	}
	
	
	void createdTask(
		void *taskObject,
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		std::lock_guard<SpinLock> guard(_graphLock);
		
		// Create the task information
		task_info_t &taskInfo = _taskToInfoMap[taskId];
		assert(taskInfo._phaseList.empty());
		
		Task *task = (Task *) taskObject;
		taskInfo._nanos6_task_info = task->getTaskInfo();
		taskInfo._nanos6_task_invocation_info = task->getTaskInvokationInfo();
		taskInfo._parent = context._taskId;
		taskInfo._status = not_created_status; // The simulation comes afterwards
		
		taskInfo._isIf0 = task->isIf0();
		
		if (context._taskId != task_id_t()) {
			task_info_t &parentInfo = _taskToInfoMap[context._taskId];
			
			parentInfo._hasChildren = true;
			
			task_group_t *taskGroup = nullptr;
			if (parentInfo._phaseList.empty()) {
				taskGroup = new task_group_t(_nextTaskwaitId++);
				parentInfo._phaseList.push_back(taskGroup);
			} else {
				phase_t *currentPhase = parentInfo._phaseList.back();
				
				taskGroup = dynamic_cast<task_group_t *> (currentPhase);
				if (taskGroup == nullptr) {
					// First task after a taskwait
					taskGroup = new task_group_t(_nextTaskwaitId++);
					parentInfo._phaseList.push_back(taskGroup);
				}
			}
			
			size_t taskGroupPhaseIndex = parentInfo._phaseList.size() - 1;
			taskInfo._taskGroupPhaseIndex = taskGroupPhaseIndex;
			
			taskGroup->_children.insert(taskId);
		}
	}
	
	void exitAddTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
}
