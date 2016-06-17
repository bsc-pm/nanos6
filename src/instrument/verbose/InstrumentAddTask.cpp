#include <atomic>
#include <cassert>

#include "InstrumentAddTask.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	task_id_t enterAddTask(nanos_task_info *taskInfo, nanos_task_invocation_info *taskInvokationInfo) {
		static std::atomic<task_id_t::inner_type_t> _nextTaskId(0);
		
		if (!_verboseAddTask) {
			return task_id_t();
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		task_id_t taskId = _nextTaskId++;
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
		}
		logEntry->_contents << " --> AddTask " << taskId;
		if (taskInfo && taskInfo->task_label) {
			logEntry->_contents << " " << taskInfo->task_label;
		}
		if (taskInvokationInfo && taskInvokationInfo->invocation_source) {
			logEntry->_contents << " " << taskInvokationInfo->invocation_source;
		}
		
		addLogEntry(logEntry);
		
		return taskId;
	}
	
	
	void createdTask(void *taskObject, task_id_t taskId) {
		if (!_verboseAddTask) {
			return;
		}
		
		Task *task = (Task *) taskObject;
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
		}
		logEntry->_contents << " --- AddTask: created " << taskId << " object:" << task;
		if (task->getParent() != nullptr) {
			logEntry->_contents << " parent:" << task->getParent()->getInstrumentationTaskId();
		}
		
		addLogEntry(logEntry);
	}
	
	
	void exitAddTask(task_id_t taskId) {
		if (!_verboseAddTask) {
			return;
		}
		
		LogEntry *logEntry = getLogEntry();
		assert(logEntry != nullptr);
		
		WorkerThread *currentWorker = WorkerThread::getCurrentWorkerThread();
		
		if (currentWorker != nullptr) {
			logEntry->_contents << "Thread:" << currentWorker << " CPU:" << currentWorker->getCpuId();
		} else {
			logEntry->_contents << "Thread:LeaderThread CPU:ANY";
		}
		logEntry->_contents << " <-- AddTask " << taskId;
		
		addLogEntry(logEntry);
	}
	
	
}
