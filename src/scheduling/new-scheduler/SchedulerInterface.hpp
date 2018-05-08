/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef SCHEDULER_INTERFACE_HPP
#define SCHEDULER_INTERFACE_HPP


#include <atomic>

#include "tasks/Task.hpp"

class SchedulerInterface {
public:
	struct polling_slot_t {
		std::atomic<Task *> _task;
		
		polling_slot_t()
			: _task(nullptr)
		{
		}
		
		Task *getTask()
		{
			Task *result = _task.load();
			while(!_task.compare_exchange_strong(result, nullptr)) {}
			return result;
		}
		
		void setTask(Task *task)
		{
			Task *expected = nullptr;
			__attribute__((unused)) bool success = _task.compare_exchange_strong(expected, task);
			assert(success);
		}
	};
	
	enum ReadyTaskHint {
		NO_HINT,
		CHILD_TASK_HINT,
		SIBLING_TASK_HINT,
		BUSY_COMPUTE_PLACE_TASK_HINT,
		UNBLOCKED_TASK_HINT,
		MAIN_TASK_HINT
	};
	
	virtual ~SchedulerInterface()
	{
	}
	
	virtual void addTaskBatch(SchedulerInterface *who, std::vector<Task *> &taskBatch) = 0;
	virtual void updateQueueThreshold(size_t queueThreshold) = 0;
};


#endif // SCHEDULER_INTERFACE_HPP
