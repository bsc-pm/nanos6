/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "TrackingPoints.hpp"
#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "monitoring/Monitoring.hpp"
#include "tasks/Task.hpp"
#include "tasks/Taskfor.hpp"

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


void TrackingPoints::taskReinitialized(Task *task)
{
	HardwareCounters::taskReinitialized(task);
	Monitoring::taskReinitialized(task);
}

void TrackingPoints::taskIsPending(Task *task)
{
	Instrument::task_id_t taskId = task->getInstrumentationTaskId();

	// No need to stop hardware counters, as the task just got created
	Instrument::taskIsPending(taskId);
}

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

void TrackingPoints::taskFinished(Task *task)
{
	// If this is a taskfor collaborator, we must accumulate its counters into the taskfor source
	if (task->isTaskforCollaborator()) {
		Taskfor *source = (Taskfor *) task->getParent();
		assert(source != nullptr);
		assert(source->isTaskfor() && source->isTaskforSource());

		// Combine the hardware counters of the taskfor collaborator (task)
		// into the taskfor source (source)
		HardwareCounters::taskCombineCounters(source, task);
	}

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
	Instrument::ThreadInstrumentationContext instrumentationContext(Instrument::task_id_t(), cpuId, threadId);
	Instrument::threadHasResumed(threadId, cpuId, false);

	HardwareCounters::threadInitialized();
}

void TrackingPoints::threadWillSuspend(const WorkerThread *thread, const CPU *cpu)
{
	assert(cpu != nullptr);
	assert(thread != nullptr);

	Instrument::thread_id_t threadId = thread->getInstrumentationId();
	Instrument::compute_place_id_t cpuId = cpu->getInstrumentationId();
	HardwareCounters::updateRuntimeCounters();
	Instrument::threadWillSuspend(threadId, cpuId);
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

	size_t id = cpu->getIndex();
	Instrument::compute_place_id_t instrumId = cpu->getInstrumentationId();

	HardwareCounters::updateRuntimeCounters();
	Monitoring::cpuBecomesIdle(id);
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

void TrackingPoints::enterWaitForIf0Task(Task *task, const Task *if0Task, const WorkerThread *thread, const CPU *cpu)
{
	assert(task != nullptr);
	assert(if0Task != nullptr);

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	const nanos6_task_invocation_info_t *if0TaskInvocation = if0Task->getTaskInvokationInfo();
	assert(if0TaskInvocation != nullptr);

	// Common function actions for when the thread suspends
	TrackingPoints::threadWillSuspend(thread, cpu);

	Instrument::enterTaskWait(taskId, if0TaskInvocation->invocation_source, if0Task->getInstrumentationTaskId(), false);
	Instrument::taskIsBlocked(taskId, Instrument::in_taskwait_blocking_reason);
	Monitoring::taskChangedStatus(task, paused_status);
}

void TrackingPoints::exitWaitForIf0Task(Task *task)
{
	assert(task != nullptr);

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();

	// We don't reset hardware counters as this is done in AddTask after
	// the waitForIf0Task function
	Instrument::taskIsExecuting(taskId, true);
	Instrument::exitTaskWait(taskId, false);

	Monitoring::taskChangedStatus(task, executing_status);
}

void TrackingPoints::enterExecuteInline(Task *task, const Task *if0Task)
{
	assert(task != nullptr);
	assert(if0Task != nullptr);

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	if (if0Task->hasCode()) {
		// Since hardware counters for the creator task (task) are updated
		// when creating the if0Task, we need not update them here
		Monitoring::taskChangedStatus(task, paused_status);
		Instrument::taskIsBlocked(taskId, Instrument::in_taskwait_blocking_reason);
	}

	const nanos6_task_invocation_info_t *if0Invocation = if0Task->getTaskInvokationInfo();
	assert(if0Invocation != nullptr);

	Instrument::enterTaskWait(taskId, if0Invocation->invocation_source, if0Task->getInstrumentationTaskId(), false);
}

void TrackingPoints::exitExecuteInline(Task *task, const Task *if0Task)
{
	assert(task != nullptr);
	assert(if0Task != nullptr);

	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	if (if0Task->hasCode()) {
		// Since hardware counters for the creator task (task) are updated
		// when creating the if0Task, we need not update them here
		Instrument::taskIsExecuting(taskId, true);
		Monitoring::taskChangedStatus(task, executing_status);
	}

	Instrument::exitTaskWait(taskId, false);
}

void TrackingPoints::enterSpawnFunction(Task *creator, bool fromUserCode)
{
	// NOTE: Our modules are interested in the transitions between Runtime and Tasks, however,
	// these functions may be called from within the runtime with "fromUserCode" set to true (e.g.
	// polling services are considered user code even if the code is from the runtime). To detect
	// these transitions, we check whether we are outside the context of a task by ensuring that
	// the current thread has a task assigned to itself. Thus, the possible scenarios are:
	//
	// 1) fromUserCode == true && currentTask != nullptr:
	//    From user code, within task context (a task calls this function). Runtime-task transition
	// 2) fromUserCode == true && currentTask == nullptr:
	//    From user code, outside task context (polling service or external thread). Not a transition
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

void TrackingPoints::enterCreateTask(Task *creator, bool fromUserCode)
{
	// NOTE: See the note in "enterSpawnFunction" for more details
	bool taskRuntimeTransition = fromUserCode && (creator != nullptr);
	if (taskRuntimeTransition) {
		HardwareCounters::updateTaskCounters(creator);
		Monitoring::taskChangedStatus(creator, paused_status);
	}
}

void TrackingPoints::enterSubmitTask(Task *creator, Task *task, bool fromUserCode)
{
	assert(task != nullptr);

	// NOTE: See the note in "enterSpawnFunction" for more details
	bool taskRuntimeTransition = fromUserCode && (creator != nullptr);
	Instrument::task_id_t taskId = task->getInstrumentationTaskId();
	Instrument::enterSubmitTask(taskRuntimeTransition);

	HardwareCounters::taskCreated(task);
	Monitoring::taskCreated(task);
	Instrument::createdTask(task, taskId);
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
