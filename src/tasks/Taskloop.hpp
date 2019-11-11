/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASKLOOP_HPP
#define TASKLOOP_HPP

#include <cmath>

#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"
#include "tasks/TaskloopInfo.hpp"

#define MOD(a, b)  ((a) < 0 ? ((((a) % (b)) + (b)) % (b)) : ((a) % (b)))

class Taskloop : public Task {
private:
	TaskloopInfo _taskloopInfo;
	
public:
	typedef nanos6_loop_bounds_t bounds_t;
	
	inline Taskloop(
		void *argsBlock, size_t argsBlockSize,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags
	)
		: Task(argsBlock, argsBlockSize, taskInfo, taskInvokationInfo, parent, instrumentationTaskId, flags, nullptr, nullptr, 0),
		_taskloopInfo()
	{}
	
	inline void reinitialize(
		void *argsBlock, size_t argsBlockSize,
		nanos6_task_info_t *taskInfo,
		nanos6_task_invocation_info_t *taskInvokationInfo,
		Task *parent,
		Instrument::task_id_t instrumentationTaskId,
		size_t flags
	)
	{
		((Task *)this)->reinitialize(argsBlock, argsBlockSize, taskInfo, taskInvokationInfo, parent, instrumentationTaskId, flags);
		_taskloopInfo.reinitialize();
	}
	
	inline TaskloopInfo const &getTaskloopInfo() const
	{
		return _taskloopInfo;
	}
	
	inline TaskloopInfo &getTaskloopInfo()
	{
		return _taskloopInfo;
	}
	
	inline void body(
		__attribute__((unused)) void *deviceEnvironment,
		__attribute__((unused)) nanos6_address_translation_entry_t *translationTable = nullptr
	) {

		nanos6_task_info_t *taskInfo = getTaskInfo();
		bool isChildTaskloop = !_taskloopInfo.isSourceTaskloop();

		if (isChildTaskloop) {
			taskInfo->implementations[0].run(getArgsBlock(), &_taskloopInfo.getBounds(), nullptr);
		}
		else {
			nanos6_task_invocation_info_t *taskInvocationInfo = getTaskInvokationInfo();
			void *originalArgsBlock = getArgsBlock();
			size_t originalArgsBlockSize = getArgsBlockSize();
			size_t flags = getFlags();

			while (hasPendingIterations()) {
				void *taskloop_ptr;
				void *argsBlock;
				Taskloop *taskloop = nullptr;

				nanos6_create_task(taskInfo, taskInvocationInfo, originalArgsBlockSize, &argsBlock, &taskloop_ptr, flags, 0);
				assert(argsBlock != nullptr);
				assert(taskloop_ptr != nullptr);

				taskloop = (Taskloop *) taskloop_ptr;

				// Copy the args block
				bool preallocatedArgsBlock = hasPreallocatedArgsBlock();
				if (!preallocatedArgsBlock) {
					if (taskInfo->duplicate_args_block != nullptr) {
						taskInfo->duplicate_args_block(originalArgsBlock, &argsBlock);
					} else {
						assert(!hasPreallocatedArgsBlock());
						memcpy(argsBlock, originalArgsBlock, originalArgsBlockSize);
					}
				}

				// Set bounds of grainsize
				bounds_t &childBounds = taskloop->getTaskloopInfo().getBounds();
				bounds_t &myBounds = _taskloopInfo.getBounds();
				childBounds.lower_bound = myBounds.lower_bound;
				myBounds.lower_bound = std::min(myBounds.lower_bound + myBounds.grainsize, myBounds.upper_bound);
				childBounds.upper_bound = myBounds.lower_bound;

				// Register deps
				nanos6_submit_task((void *)taskloop);

				// Instrument the task creation
				//Instrument::task_id_t taskInstrumentationId = taskloop->getInstrumentationTaskId();
				//Instrument::createdTask(taskloop, taskInstrumentationId);
				//Instrument::exitAddTask(taskInstrumentationId);
			}
		}
	}
	
	inline void setRunnable(bool runnableValue)
	{
		_flags[Task::non_runnable_flag] = !runnableValue;
	}
	
	inline bool hasPendingIterations()
	{
		return (_taskloopInfo.getIterationCount() > 0);
	}

	inline bool isSourceTaskloop()
	{
		return _taskloopInfo.isSourceTaskloop();
	}
};

#endif // TASKLOOP_HPP
