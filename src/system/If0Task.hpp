/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef IF0_TASK_HPP
#define IF0_TASK_HPP

#include <cassert>

#include "executors/threads/CPU.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"

#include <HardwareCounters.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentTaskWait.hpp>
#include <Monitoring.hpp>


class ComputePlace;


namespace If0Task {
	inline void waitForIf0Task(WorkerThread *currentThread, Task *currentTask, Task *if0Task, ComputePlace *computePlace)
	{
		assert(currentThread != nullptr);
		assert(currentTask != nullptr);
		assert(if0Task != nullptr);
		assert(computePlace != nullptr);
		
		CPU *cpu = static_cast<CPU *>(computePlace);
		
		Instrument::enterTaskWait(currentTask->getInstrumentationTaskId(), if0Task->getTaskInvokationInfo()->invocation_source, if0Task->getInstrumentationTaskId());
		
		WorkerThread *replacementThread = ThreadManager::getIdleThread(cpu);
		
		Monitoring::taskChangedStatus(currentTask, blocked_status, cpu);
		HardwareCounters::stopTaskMonitoring(currentTask);
		
		Instrument::taskIsBlocked(currentTask->getInstrumentationTaskId(), Instrument::in_taskwait_blocking_reason);
		currentThread->switchTo(replacementThread);
		
		//Update the CPU since the thread may have migrated
		cpu = currentThread->getComputePlace();
		assert(cpu != nullptr);
		Instrument::ThreadInstrumentationContext::updateComputePlace(cpu->getInstrumentationId());
		
		Instrument::exitTaskWait(currentTask->getInstrumentationTaskId());
		Instrument::taskIsExecuting(currentTask->getInstrumentationTaskId());
		
		assert(currentTask->getThread() != nullptr);
		HardwareCounters::startTaskMonitoring(currentTask);
		Monitoring::taskChangedStatus(currentTask, executing_status, cpu);
	}
	
	
	inline void executeInline(
		WorkerThread *currentThread, Task *currentTask, Task *if0Task,
		ComputePlace *computePlace
	) {
		assert(currentThread != nullptr);
		assert(currentTask != nullptr);
		assert(if0Task != nullptr);
		assert(if0Task->getParent() == currentTask);
		assert(computePlace != nullptr);
		
		bool hasCode = if0Task->hasCode();
		
		Instrument::enterTaskWait(currentTask->getInstrumentationTaskId(), if0Task->getTaskInvokationInfo()->invocation_source, if0Task->getInstrumentationTaskId());
		if (hasCode) {
			Monitoring::taskChangedStatus(currentTask, blocked_status, computePlace);
			HardwareCounters::stopTaskMonitoring(currentTask);
			
			Instrument::taskIsBlocked(currentTask->getInstrumentationTaskId(), Instrument::in_taskwait_blocking_reason);
		}
		
		currentThread->handleTask((CPU *) computePlace, if0Task);
		
		Instrument::exitTaskWait(currentTask->getInstrumentationTaskId());
		
		if (hasCode) {
			Instrument::taskIsExecuting(currentTask->getInstrumentationTaskId());
			
			assert(currentTask->getThread() != nullptr);
			HardwareCounters::startTaskMonitoring(currentTask);
			Monitoring::taskChangedStatus(currentTask, executing_status, currentTask->getThread()->getComputePlace());
		}
	}
	
	
	inline void executeNonInline(
		WorkerThread *currentThread, Task *if0Task,
		ComputePlace *computePlace
	) {
		assert(currentThread != nullptr);
		assert(if0Task != nullptr);
		assert(computePlace != nullptr);
		
		assert(if0Task->isIf0());
		
		Task *parent = if0Task->getParent();
		assert(parent != nullptr);
		
		currentThread->handleTask((CPU *) computePlace, if0Task);
		
		// The thread can migrate during the execution of the task
		computePlace = currentThread->getComputePlace();
		
		Scheduler::addReadyTask(parent, computePlace, SchedulerInterface::UNBLOCKED_TASK_HINT);
	}
	
}


#endif // IF0_TASK_HPP
