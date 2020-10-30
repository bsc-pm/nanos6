/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef METRIC_POINTS_HPP
#define METRIC_POINTS_HPP

#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "tasks/Task.hpp"

#include <InstrumentAddTask.hpp>
#include <InstrumentBlockingAPI.hpp>
#include <InstrumentSchedulerSubsystemEntryPoints.hpp>
#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentTaskWait.hpp>
#include <InstrumentThreadManagement.hpp>
#include <InstrumentUserMutex.hpp>
#include <Monitoring.hpp>


//! \brief This namespace aggregates common Instrumentation, Monitoring, and
//! HardwareCounter actions in order to simplify the runtime core
namespace MetricPoints {

	//    COMMON FLOW OF TASKS/THREADS/CPUS    //

	inline void taskReinitialized(Task *task)
	{
		HardwareCounters::taskReinitialized(task);
		Monitoring::taskReinitialized(this);
	}

	inline void taskIsPending(Instrument::task_id_t taskId)
	{
		// No need to stop hardware counters, as the task just got created
		Instrument::taskIsPending(taskId);
	}

	void taskIsExecuting(Task *task);

	void taskCompletedUserCode(Task *task, bool taskHasCode);

	inline void taskFinished(Task *task)
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

	inline void threadWillSuspend(Instrument::thread_id_t threadId, Instrument::compute_place_id_t cpuId)
	{
		HardwareCounters::updateRuntimeCounters();
		Instrument::threadWillSuspend(threadId, cpuId);
	}

	inline void threadWillShutdown()
	{
		HardwareCounters::updateRuntimeCounters();
		Instrument::threadWillShutdown();
		HardwareCounters::threadShutdown();
	}


	//    ENTRY/EXIT POINTS OF THE RUNTIME CORE    //

	inline void enterTaskWait(
		Task *task, Instrument::task_id_t taskId,
		char const *invocationSource, bool fromUserCode
	) {
		if (fromUserCode) {
			HardwareCounters::updateTaskCounters(task);
			Monitoring::taskChangedStatus(task, paused_status);
		}

		Instrument::enterTaskWait(taskId, invocationSource, Instrument::task_id_t(), fromUserCode);
	}

	inline void exitTaskWait(Task *task, Instrument::task_id_t taskId, bool fromUserCode)
	{
		if (fromUserCode) {
			HardwareCounters::updateRuntimeCounters();
			Instrument::exitTaskWait(taskId, fromUserCode);
			Monitoring::taskChangedStatus(task, executing_status);
		} else {
			Instrument::exitTaskWait(taskId, fromUserCode);
		}
	}

	inline void enterWaitForIf0Task(
		Task *task, Instrument::task_id_t taskId, char const *invocationSource,
		Instrument::task_id_t if0TaskId, WorkerThread *thread, CPU *cpu
	) {
		assert(cpu != nullptr);
		assert(thread != nullptr);

		// Common function actions for when the thread suspends
		MetricPoints::threadWillSuspend(thread->getInstrumentationId(), cpu->getInstrumentationId());

		Instrument::enterTaskWait(taskId, invocationSource, if0TaskId, false);
		Instrument::taskIsBlocked(taskId, Instrument::in_taskwait_blocking_reason);
		Monitoring::taskChangedStatus(task, paused_status);
	}

	inline void exitWaitForIf0Task(Task *task, Instrument::task_id_t taskId)
	{
		// We don't reset hardware counters as this is done in AddTask after
		// the waitForIf0Task function
		Instrument::taskIsExecuting(taskId, true);
		Instrument::exitTaskWait(taskId, false);

		Monitoring::taskChangedStatus(task, executing_status);
	}

	inline void enterExecuteInline(
		Task *task, Instrument::task_id_t taskId, char const *invocationSource,
		Instrument::task_id_t if0TaskId, bool hasCode
	) {
		if (hasCode) {
			// Since hardware counters for the creator task (task) are updated
			// when creating the if0Task, we need not update them here
			Monitoring::taskChangedStatus(task, paused_status);
			Instrument::taskIsBlocked(taskId, Instrument::in_taskwait_blocking_reason);
		}

		Instrument::enterTaskWait(taskId, invocationSource, if0TaskId, false);
	}

	inline void exitExecuteInline(Task *task, Instrument::task_id_t taskId, bool hasCode)
	{
		if (hasCode) {
			// Since hardware counters for the creator task (task) are updated
			// when creating the if0Task, we need not update them here
			Instrument::taskIsExecuting(taskId, true);
			Monitoring::taskChangedStatus(task, executing_status);
		}

		Instrument::exitTaskWait(taskId, false);
	}

	inline void enterSpawnFunction(Task *creator, bool fromUserCode)
	{
		if (fromUserCode) {
			HardwareCounters::updateTaskCounters(creator);
			Monitoring::taskChangedStatus(creator, paused_status);
		}

		Instrument::enterSpawnFunction(fromUserCode);
	}

	inline void exitSpawnFunction(Task *creator, bool fromUserCode)
	{
		if (fromUserCode) {
			HardwareCounters::updateRuntimeCounters();
			Instrument::exitSpawnFunction(fromUserCode);
			Monitoring::taskChangedStatus(creator, executing_status);
		} else {
			Instrument::exitSpawnFunction(fromUserCode);
		}
	}

	inline void enterUserLock(Task *task)
	{
		HardwareCounters::updateTaskCounters(task);
		Monitoring::taskChangedStatus(task, paused_status);
		Instrument::enterUserMutexLock();
	}

	inline void exitUserLock(Task *task)
	{
		HardwareCounters::updateRuntimeCounters();
		Instrument::exitUserMutexLock();
		Monitoring::taskChangedStatus(task, executing_status);
	}

	inline void enterUserUnlock(Task *task)
	{
		HardwareCounters::updateTaskCounters(task);
		Monitoring::taskChangedStatus(task, paused_status);
		Instrument::enterUserMutexUnlock();
	}

	inline void exitUserUnlock(Task *task)
	{
		HardwareCounters::updateRuntimeCounters();
		Instrument::exitUserMutexUnlock();
		Monitoring::taskChangedStatus(task, executing_status);
	}

	inline void enterBlockCurrentTask(Task *task, Instrument::task_id_t taskId, bool fromUserCode)
	{
		if (fromUserCode) {
			HardwareCounters::updateTaskCounters(task);
			Monitoring::taskChangedStatus(task, paused_status);
		}
		Instrument::enterBlockCurrentTask(taskId, fromUserCode);
		Instrument::taskIsBlocked(taskId, Instrument::user_requested_blocking_reason);
	}

	inline void exitBlockCurrentTask(Task *task, Instrument::task_id_t taskId, bool fromUserCode)
	{
		Instrument::taskIsExecuting(taskId, true);
		if (fromUserCode) {
			HardwareCounters::updateRuntimeCounters();
			Instrument::exitBlockCurrentTask(taskId, fromUserCode);
			Monitoring::taskChangedStatus(task, executing_status);
		} else {
			Instrument::exitBlockCurrentTask(taskId, fromUserCode);
		}
	}


	inline void enterUnblockCurrentTask(Task *task, Instrument::task_id_t taskId, bool fromUserCode)
	{
		if (fromUserCode) {
			HardwareCounters::updateTaskCounters(task);
			Monitoring::taskChangedStatus(task, paused_status);
		}
		Instrument::enterUnblockTask(taskId, fromUserCode);
	}

	inline void exitUnblockCurrentTask(Task *task, Instrument::task_id_t taskId, bool fromUserCode)
	{
		if (fromUserCode) {
			HardwareCounters::updateRuntimeCounters();
			Instrument::exitUnblockTask(taskId, fromUserCode);
			Monitoring::taskChangedStatus(task, executing_status);
		} else {
			Instrument::exitUnblockTask(taskId, fromUserCode);
		}
	}

	inline void enterWaitFor(Task *task, Instrument::task_id_t taskId)
	{
		HardwareCounters::updateTaskCounters(task);
		// We do not notify Monitoring yet, as this will be done in the Scheduler's addReadyTask call
		Instrument::enterWaitFor(taskId);
	}

	inline void exitWaitFor(Task *task, Instrument::task_id_t taskId)
	{
		HardwareCounters::updateRuntimeCounters();
		Instrument::exitWaitFor(taskId);
		Monitoring::taskChangedStatus(task, executing_status);
	}

	inline void enterAddReadyTasks(Task *tasks[], const size_t numTasks)
	{
		Instrument::enterAddReadyTask();
		for (size_t i = 0; i < numTasks; ++i) {
			Task *task = tasks[i];
			assert(task != nullptr);

			Instrument::taskIsReady(task->getInstrumentationTaskId());
			Monitoring::taskChangedStatus(task, ready_status);
		}
	}

	inline void exitAddReadyTasks()
	{
		Instrument::exitAddReadyTask();
	}

	inline void enterAddReadyTask(Task *task)
	{
		Instrument::enterAddReadyTask();
		Instrument::taskIsReady(task->getInstrumentationTaskId());
		Monitoring::taskChangedStatus(task, ready_status);
	}

	inline void exitAddReadyTask()
	{
		Instrument::exitAddReadyTask();
	}

	inline void enterCreateTask(Task *creator, bool fromUserCode)
	{
		if (fromUserCode) {
			HardwareCounters::updateTaskCounters(creator);
			Monitoring::taskChangedStatus(creator, paused_status);
		}
	}

	inline void enterSubmitTask(Task *task, Instrument::task_id_t taskId, bool fromUserCode)
	{
		Instrument::enterSubmitTask(fromUserCode);

		HardwareCounters::taskCreated(task);
		Monitoring::taskCreated(task);
		Instrument::createdTask(task, taskId);
	}

	inline void exitSubmitTask(Task *creator, Instrument::task_id_t taskId, bool fromUserCode)
	{
		if (fromUserCode) {
			HardwareCounters::updateRuntimeCounters();
			Instrument::exitSubmitTask(taskId, fromUserCode);
			Monitoring::taskChangedStatus(creator, executing_status);
		} else {
			Instrument::exitSubmitTask(taskId, fromUserCode);
		}
	}

}

#endif // METRIC_POINTS_HPP
