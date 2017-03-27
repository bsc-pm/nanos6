#include "InstrumentAddTask.hpp"

#include "ExecutionSteps.hpp"
#include "InstrumentGraph.hpp"

#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"

#include <cassert>
#include <mutex>


namespace Instrument {
	using namespace Graph;
	
	
	task_id_t enterAddTask(
		__attribute__((unused)) nanos_task_info *taskInfo,
		__attribute__((unused)) nanos_task_invocation_info *taskInvokationInfo,
		__attribute__((unused)) size_t flags
	) {
		std::lock_guard<SpinLock> guard(_graphLock);
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread == nullptr) {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		thread_id_t threadId = 0;
		if (currentThread != nullptr) {
			threadId = currentThread->getInstrumentationId();
		}
		
		// Get an ID for the task
		task_id_t taskId = _nextTaskId++;
		
		long cpuId = -2;
		if (currentThread != nullptr) {
			CPU *cpu = currentThread->getHardwarePlace();
			assert(cpu != nullptr);
			cpuId = cpu->_virtualCPUId;
		}
		
		// Add the event before the task to avoid ordering issues during the simulation
		task_id_t parentTaskId = -1;
		if (currentThread != nullptr) {
			Task *parentTask = currentThread->getTask();
			if (parentTask != nullptr) {
				parentTaskId = parentTask->getInstrumentationTaskId();
			}
		}
		
		create_task_step_t *createTaskStep = new create_task_step_t(cpuId, threadId, taskId, parentTaskId);
		_executionSequence.push_back(createTaskStep);
		
		return taskId;
	}
	
	
	void createdTask(void *taskObject, task_id_t taskId)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		Task *parentTask = nullptr;
		
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		if (currentThread != nullptr) {
			parentTask = currentThread->getTask();
			assert(parentTask != nullptr);
		} else {
			// The main task gets added by a non-worker thread
			// And other tasks can also be added by external threads in library mode
		}
		
		// Create the task information
		task_info_t &taskInfo = _taskToInfoMap[taskId];
		assert(taskInfo._phaseList.empty());
		
		Task *task = (Task *) taskObject;
		taskInfo._nanos_task_info = task->getTaskInfo();
		taskInfo._nanos_task_invocation_info = task->getTaskInvokationInfo();
		if (parentTask != 0) {
			taskInfo._parent = parentTask->getInstrumentationTaskId();
		}
		taskInfo._status = not_created_status; // The simulation comes afterwards
		
		if (parentTask != nullptr) {
			task_id_t parentTaskId = parentTask->getInstrumentationTaskId();
			task_info_t &parentInfo = _taskToInfoMap[parentTaskId];
			
			parentInfo._hasChildren = true;
			
			task_group_t *taskGroup = nullptr;
			if (!parentInfo._phaseList.empty()) {
				phase_t *lastPhase = parentInfo._phaseList.back();
				taskGroup = dynamic_cast<task_group_t *>(lastPhase);
			}
			
			if (taskGroup == nullptr) {
				taskGroup = new task_group_t(_nextTaskwaitId++);
				parentInfo._phaseList.push_back(taskGroup);
			}
			assert(taskGroup != nullptr);
			
			taskGroup->_children.insert(taskId);
		}
	}
	
	void exitAddTask(__attribute__((unused)) task_id_t taskId)
	{
	}
	
}
