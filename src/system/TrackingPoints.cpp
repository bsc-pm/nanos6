/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#include "TrackingPoints.hpp"
#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "monitoring/Monitoring.hpp"
#include "tasks/Task.hpp"

#include <InstrumentAddTask.hpp>
#include <InstrumentBlockingAPI.hpp>
#include <InstrumentComputePlaceManagement.hpp>
#include <InstrumentScheduler.hpp>
#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentTaskWait.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>
#include <InstrumentThreadManagement.hpp>
#include <InstrumentUserMutex.hpp>


void TrackingPoints::taskIsPending(const Task *task)
{
	assert(task != nullptr);

	// No need to stop hardware counters, as the task just got created
	Instrument::taskIsPending(task->getInstrumentationTaskId());
}

void TrackingPoints::taskIsExecuting(Task *task)
{
	assert(task != nullptr);

	HardwareCounters::updateRuntimeCounters();

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	Instrument::startTask(taskId);
	Instrument::taskIsExecuting(taskId);

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
		Instrument::taskIsZombie(taskId);
		Instrument::endTask(taskId);
	} else {
		Monitoring::taskChangedStatus(task, paused_status);
		Monitoring::taskCompletedUserCode(task);
	}
}

void TrackingPoints::taskFinished(Task *task)
{
	assert(task != nullptr);

	// Propagate monitoring actions for this task since it has finished
	Monitoring::taskFinished(task);
}

void TrackingPoints::threadInitialized(const WorkerThread *thread, const CPU *cpu)
{
	assert(thread != nullptr);
	assert(cpu != nullptr);

	Instrument::thread_id_t threadId = thread->getInstrumentationId();
	Instrument::compute_place_id_t cpuId = cpu->getInstrumentationId();
	Instrument::createdThread(threadId, cpuId);
	Instrument::threadHasResumed(threadId, cpuId, false);

	HardwareCounters::threadInitialized();
}

void TrackingPoints::threadWillSuspend(const WorkerThread *thread, const CPU *cpu)
{
	assert(cpu != nullptr);
	assert(thread != nullptr);

	HardwareCounters::updateRuntimeCounters();
	Instrument::threadWillSuspend(thread->getInstrumentationId(), cpu->getInstrumentationId());
}

void TrackingPoints::threadWillShutdown()
{
	HardwareCounters::updateRuntimeCounters();
	Instrument::threadWillShutdown();
	HardwareCounters::threadShutdown();
}

void TrackingPoints::cpuBecomesActive(const CPU *cpu)
{
	assert(cpu != nullptr);

	Instrument::resumedComputePlace(cpu->getInstrumentationId());
	Monitoring::cpuBecomesActive(cpu->getIndex());
}

void TrackingPoints::cpuBecomesIdle(const CPU *cpu, const WorkerThread *thread)
{
	assert(cpu != nullptr);
	assert(thread != nullptr);

	Instrument::compute_place_id_t instrumId = cpu->getInstrumentationId();

	HardwareCounters::updateRuntimeCounters();
	Monitoring::cpuBecomesIdle(cpu->getIndex());
	Instrument::threadWillSuspend(thread->getInstrumentationId(), instrumId);
	Instrument::suspendingComputePlace(instrumId);
}


//    ENTRY/EXIT POINTS OF THE RUNTIME CORE    //

void TrackingPoints::enterTaskWait(Task *task, char const *invocationSource, bool fromUserCode)
{
	assert(task != nullptr);

	if (fromUserCode) {
		HardwareCounters::updateTaskCounters(task);
		Monitoring::taskChangedStatus(task, paused_status);
	}

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	Instrument::enterTaskWait(taskId, invocationSource, Instrument::task_id_t(), fromUserCode);
}

void TrackingPoints::exitTaskWait(Task *task, bool fromUserCode)
{
	assert(task != nullptr);

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	if (fromUserCode) {
		HardwareCounters::updateRuntimeCounters();
		Instrument::exitTaskWait(taskId, fromUserCode);
		Monitoring::taskChangedStatus(task, executing_status);
	} else {
		Instrument::exitTaskWait(taskId, fromUserCode);
	}
}

void TrackingPoints::enterWaitForIf0Task(const Task *task, const Task *if0Task, const WorkerThread *thread, const CPU *cpu)
{
	assert(task != nullptr);
	assert(if0Task != nullptr);
	assert(thread != nullptr);
	assert(cpu != nullptr);

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	const nanos6_task_invocation_info_t *if0TaskInvocation = if0Task->getTaskInvokationInfo();
	assert(if0TaskInvocation != nullptr);

	Instrument::enterTaskWait(taskId, if0TaskInvocation->invocation_source, if0Task->getInstrumentationTaskId(), false);
	Instrument::taskIsBlocked(taskId, Instrument::in_taskwait_blocking_reason);

	HardwareCounters::updateRuntimeCounters();
	Instrument::threadWillSuspend(thread->getInstrumentationId(), cpu->getInstrumentationId());
}

void TrackingPoints::exitWaitForIf0Task(const Task *task)
{
	assert(task != nullptr);

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	Instrument::taskIsExecuting(taskId, true);
	Instrument::exitTaskWait(taskId, false);
}

void TrackingPoints::enterExecuteInline(const Task *task, const Task *if0Task)
{
	assert(task != nullptr);
	assert(if0Task != nullptr);

	const nanos6_task_invocation_info_t *if0Invocation = if0Task->getTaskInvokationInfo();
	assert(if0Invocation != nullptr);

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	Instrument::enterTaskWait(taskId, if0Invocation->invocation_source, if0Task->getInstrumentationTaskId(), false);
	if (if0Task->hasCode()) {
		Instrument::taskIsBlocked(taskId, Instrument::in_taskwait_blocking_reason);
	}
}

void TrackingPoints::exitExecuteInline(const Task *task, const Task *if0Task)
{
	assert(task != nullptr);
	assert(if0Task != nullptr);

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	if (if0Task->hasCode()) {
		// Since hardware counters for the creator task (task) are updated
		// when creating the if0Task, we need not update them here
		Instrument::taskIsExecuting(taskId, true);
	}

	Instrument::exitTaskWait(taskId, false);
}

void TrackingPoints::enterSpawnFunction(Task *creator, bool fromUserCode)
{
	// NOTE: Our modules are interested in the transitions between Runtime and Tasks, however,
	// these functions may be called from within the runtime with "fromUserCode" set to true. To
	// detect these transitions, we check whether we are outside the context of a task by ensuring
	// that the current thread has a task assigned to itself. Thus, the possible scenarios are:
	//
	// 1) fromUserCode == true && currentTask != nullptr:
	//    From user code, within task context (a task calls this function). Runtime-task transition
	// 2) fromUserCode == true && currentTask == nullptr:
	//    From user code, outside task context (external thread). Not a transition
	// 3) fromUserCode == false:
	//    From runtime code. This is not a transition between runtime and task context.

	bool taskRuntimeTransition = fromUserCode && (creator != nullptr);
	if (taskRuntimeTransition) {
		HardwareCounters::updateTaskCounters(creator);
		Monitoring::taskChangedStatus(creator, paused_status);
	}

	Instrument::enterSpawnFunction(taskRuntimeTransition);
}

void TrackingPoints::exitSpawnFunction(Task *creator, bool fromUserCode)
{
	// NOTE: See the note in "enterSpawnFunction" for more details
	bool taskRuntimeTransition = fromUserCode && (creator != nullptr);
	if (taskRuntimeTransition) {
		HardwareCounters::updateRuntimeCounters();
		Instrument::exitSpawnFunction(taskRuntimeTransition);
		Monitoring::taskChangedStatus(creator, executing_status);
	} else {
		Instrument::exitSpawnFunction(taskRuntimeTransition);
	}
}

void TrackingPoints::enterUserLock(Task *task)
{
	HardwareCounters::updateTaskCounters(task);
	Monitoring::taskChangedStatus(task, paused_status);
	Instrument::enterUserMutexLock();
}

void TrackingPoints::exitUserLock(Task *task)
{
	HardwareCounters::updateRuntimeCounters();
	Instrument::exitUserMutexLock();
	Monitoring::taskChangedStatus(task, executing_status);
}

void TrackingPoints::enterUserUnlock(Task *task)
{
	HardwareCounters::updateTaskCounters(task);
	Monitoring::taskChangedStatus(task, paused_status);
	Instrument::enterUserMutexUnlock();
}

void TrackingPoints::exitUserUnlock(Task *task)
{
	HardwareCounters::updateRuntimeCounters();
	Instrument::exitUserMutexUnlock();
	Monitoring::taskChangedStatus(task, executing_status);
}

void TrackingPoints::enterBlockCurrentTask(Task *task, bool fromUserCode)
{
	assert(task != nullptr);

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	if (fromUserCode) {
		HardwareCounters::updateTaskCounters(task);
		Monitoring::taskChangedStatus(task, paused_status);
	}
	Instrument::enterBlockCurrentTask(taskId, fromUserCode);
	Instrument::taskIsBlocked(taskId, Instrument::user_requested_blocking_reason);
}

void TrackingPoints::exitBlockCurrentTask(Task *task, bool fromUserCode)
{
	assert(task != nullptr);

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	Instrument::taskIsExecuting(taskId, true);
	if (fromUserCode) {
		HardwareCounters::updateRuntimeCounters();
		Instrument::exitBlockCurrentTask(taskId, fromUserCode);
		Monitoring::taskChangedStatus(task, executing_status);
	} else {
		Instrument::exitBlockCurrentTask(taskId, fromUserCode);
	}
}

void TrackingPoints::enterUnblockTask(const Task *task, Task *currentTask, bool fromUserCode)
{
	// NOTE: See the note in "enterSpawnFunction" for more details
	assert(task != nullptr);

	bool taskRuntimeTransition = fromUserCode && (currentTask != nullptr);
	if (taskRuntimeTransition) {
		HardwareCounters::updateTaskCounters(currentTask);
		Monitoring::taskChangedStatus(currentTask, paused_status);
	}

	Instrument::enterUnblockTask(task->getInstrumentationTaskId(), taskRuntimeTransition);
}

void TrackingPoints::exitUnblockTask(const Task *task, Task *currentTask, bool fromUserCode)
{
	// NOTE: See the note in "enterSpawnFunction" for more details
	assert(task != nullptr);

	bool taskRuntimeTransition = fromUserCode && (currentTask != nullptr);
	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	if (taskRuntimeTransition) {
		HardwareCounters::updateRuntimeCounters();
		Instrument::exitUnblockTask(taskId, taskRuntimeTransition);
		Monitoring::taskChangedStatus(currentTask, executing_status);
	} else {
		Instrument::exitUnblockTask(taskId, taskRuntimeTransition);
	}
}

void TrackingPoints::enterWaitFor(Task *task)
{
	assert(task != nullptr);

	HardwareCounters::updateTaskCounters(task);
	// We do not notify Monitoring yet, as this will be done in the Scheduler's addReadyTask call
	Instrument::enterWaitFor(task->getInstrumentationTaskId());
}

void TrackingPoints::exitWaitFor(Task *task)
{
	assert(task != nullptr);

	HardwareCounters::updateRuntimeCounters();
	Instrument::exitWaitFor(task->getInstrumentationTaskId());
	Monitoring::taskChangedStatus(task, executing_status);
}

void TrackingPoints::enterAddReadyTasks(Task *tasks[], const size_t numTasks)
{
	Instrument::enterAddReadyTask();

	for (size_t i = 0; i < numTasks; ++i) {
		Task *task = tasks[i];
		assert(task != nullptr);

		Instrument::taskIsReady(task->getInstrumentationTaskId());
		Monitoring::taskChangedStatus(task, ready_status);
	}
}

void TrackingPoints::exitAddReadyTasks()
{
	Instrument::exitAddReadyTask();
}

void TrackingPoints::enterAddReadyTask(Task *task)
{
	assert(task != nullptr);

	Instrument::enterAddReadyTask();
	Instrument::taskIsReady(task->getInstrumentationTaskId());
	Monitoring::taskChangedStatus(task, ready_status);
}

void TrackingPoints::exitAddReadyTask()
{
	Instrument::exitAddReadyTask();
}

Instrument::task_id_t TrackingPoints::enterCreateTask(
	Task *creator,
	nanos6_task_info_t *taskInfo,
	nanos6_task_invocation_info_t *taskInvocationInfo,
	size_t flags,
	bool fromUserCode
) {
	// NOTE: See the note in "enterSpawnFunction" for more details
	bool taskRuntimeTransition = fromUserCode && (creator != nullptr);
	if (taskRuntimeTransition) {
		HardwareCounters::updateTaskCounters(creator);
		Monitoring::taskChangedStatus(creator, paused_status);
	}

	return Instrument::enterCreateTask(taskInfo, taskInvocationInfo, flags, taskRuntimeTransition);
}

void TrackingPoints::exitCreateTask(const Task *creator, bool fromUserCode)
{
	// NOTE: See the note in "enterSpawnFunction" for more details
	bool taskRuntimeTransition = fromUserCode && (creator != nullptr);
	Instrument::exitCreateTask(taskRuntimeTransition);
}

void TrackingPoints::enterSubmitTask(const Task *creator, Task *task, bool fromUserCode)
{
	assert(task != nullptr);

	// NOTE: See the note in "enterSpawnFunction" for more details
	bool taskRuntimeTransition = fromUserCode && (creator != nullptr);
	Instrument::enterSubmitTask(taskRuntimeTransition);

	HardwareCounters::taskCreated(task);
	Monitoring::taskCreated(task);
	Instrument::createdTask(task, task->getInstrumentationTaskId());
}

void TrackingPoints::exitSubmitTask(Task *creator, const Task *task, bool fromUserCode)
{
	assert(task != nullptr);

	// NOTE: See the note in "enterSpawnFunction" for more details
	bool taskRuntimeTransition = fromUserCode && (creator != nullptr);
	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	if (taskRuntimeTransition) {
		HardwareCounters::updateRuntimeCounters();
		Instrument::exitSubmitTask(taskId, taskRuntimeTransition);
		Monitoring::taskChangedStatus(creator, executing_status);
	} else {
		Instrument::exitSubmitTask(taskId, taskRuntimeTransition);
	}
}
