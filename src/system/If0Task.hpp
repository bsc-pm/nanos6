#ifndef IF0_TASK_HPP
#define IF0_TASK_HPP

#include <cassert>

#include <InstrumentTaskWait.hpp>
#include <InstrumentTaskStatus.hpp>

#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"


class HardwarePlace;


namespace If0Task {
	inline void waitForIf0Task(WorkerThread *currentThread, Task *currentTask, Task *if0Task, HardwarePlace *hardwarePlace)
	{
		assert(currentThread != nullptr);
		assert(currentTask != nullptr);
		assert(if0Task != nullptr);
		assert(hardwarePlace != nullptr);
		
		CPU *cpu = static_cast<CPU *>(hardwarePlace);
		
		Instrument::enterTaskWait(currentTask->getInstrumentationTaskId(), if0Task->getTaskInvokationInfo()->invocation_source);
		
		WorkerThread *replacementThread = ThreadManager::getIdleThread(cpu);
		
		Instrument::taskIsBlocked(currentTask->getInstrumentationTaskId(), Instrument::in_taskwait_blocking_reason);
		ThreadManager::switchThreads(currentThread, replacementThread);
		
		Instrument::exitTaskWait(currentTask->getInstrumentationTaskId());
		Instrument::taskIsExecuting(currentTask->getInstrumentationTaskId());
	}
	
	
	inline void executeInline(WorkerThread *currentThread, Task *currentTask, Task *if0Task)
	{
		assert(currentThread != nullptr);
		assert(currentTask != nullptr);
		assert(if0Task != nullptr);
		assert(if0Task->getParent() == currentTask);
		
		Instrument::enterTaskWait(currentTask->getInstrumentationTaskId(), if0Task->getTaskInvokationInfo()->invocation_source);
		if (if0Task->hasCode()) {
			Instrument::taskIsBlocked(currentTask->getInstrumentationTaskId(), Instrument::in_taskwait_blocking_reason);
			currentThread->handleTask(if0Task);
		}
		Instrument::exitTaskWait(currentTask->getInstrumentationTaskId());
		
		if (if0Task->hasCode()) {
			Instrument::taskIsExecuting(currentTask->getInstrumentationTaskId());
		}
	}
	
	
	inline void executeNonInline(WorkerThread *currentThread, Task *if0Task, HardwarePlace *hardwarePlace)
	{
		assert(currentThread != nullptr);
		assert(if0Task != nullptr);
		assert(hardwarePlace != nullptr);
		
		assert(if0Task->isIf0());
		
		Task *parent = if0Task->getParent();
		assert(parent != nullptr);
		
		if (if0Task->hasCode()) {
			currentThread->handleTask(if0Task);
		}
		
		Scheduler::taskGetsUnblocked(parent, hardwarePlace);
	}
	
}


#endif // IF0_TASK_HPP
