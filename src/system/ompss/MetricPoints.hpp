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
#include <InstrumentScheduler.hpp>
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

	//! \brief Actions to be taken after a task is reinitialized (commonly Taskfors)
	//!
	//! Actions:
	//! - HWCounters: Notify that the current task has reinitialized
	//! - Monitoring: Notify that the current task has reinitialized
	//!
	//! \param[in] task The reinitialized task
	inline void taskReinitialized(Task *task)
	{
		HardwareCounters::taskReinitialized(task);
		Monitoring::taskReinitialized(this);
	}

	//! \brief Actions to be taken when a task switches to pending status
	//!
	//! Actions:
	//! - Instrument: Notify that the task is pending
	//!
	//! \param[in] taskId The task's instrumentation id
	inline void taskIsPending(Instrument::task_id_t taskId)
	{
		// No need to stop hardware counters, as the task just got created
		Instrument::taskIsPending(taskId);
	}

	//! \brief Actions to be taken after a task begins executing user code
	//!
	//! Actions:
	//! - HWCounters: Update runtime counters since the task is about to execute
	//! - Instrument: Notify that a task is about to begin executing (or a taskfor)
	//! - Monitoring: Notify that a task is about to begin executing
	//!
	//! \param[in] task The task about to execute
	void taskIsExecuting(Task *task);

	//! \brief Actions to be taken after a task has completed user code execution
	//!
	//! Actions:
	//! - HWCounters: Update and accumulate task counters since the task has finished executing
	//! - Instrument: Notify that a task has finished user code execution
	//! - Monitoring: Notify that a task has finished user code execution
	//!
	//! \param[in] task The task that has completed its user code
	//! \param[in] taskHasCode Whether the task had code
	void taskCompletedUserCode(Task *task, bool taskHasCode);

	//! \brief Actions to be taken after a task has completely finished, meaning
	//! the tasks and all its children have completely finished their execution
	//!
	//! Actions:
	//! - HWCounters: If the task is a taskfor collaborator, combine its counters
	//!   to the taskfor source
	//! - Monitoring: Notify that the current task has completely finished
	//!
	//! \param[in] task The finished task
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

	//! \brief Common actions within the namespace for when a thread is suspending
	//!
	//! Actions:
	//! - HWCounters: Update runtime counters as this thread will be suspended
	//!   and it may migrate from CPU
	//! - Instrument: Notify that the current thread will suspend
	//!
	//! \param[in] threadId The id of the (soon to be suspended) current thread
	//! \param[in] cpu The id of the current CPU which the thread is running on
	inline void threadWillSuspend(Instrument::thread_id_t threadId, Instrument::compute_place_id_t cpuId)
	{
		HardwareCounters::updateRuntimeCounters();
		Instrument::threadWillSuspend(threadId, cpuId);
	}

	//! \brief Actions to take after a thread is shutting down
	//!
	//! Actions:
	//! - HWCounters: Update runtime counters for the thread and shutdown counter reading
	//! - Instrument: Notify that the thread is shutting down
	//!
	inline void threadWillShutdown()
	{
		HardwareCounters::updateRuntimeCounters();
		Instrument::threadWillShutdown();
		HardwareCounters::threadShutdown();
	}


	//    ENTRY/EXIT POINTS OF THE RUNTIME CORE    //

	//! \brief Entry point of the "taskWait" function (TaskWait.cpp)
	//!
	//! Actions:
	//! - HWCounters: Update (read) the task's counters as it will be blocked
	//! - Monitoring: Notify that the task becomes blocked
	//! - Instrument: Notify that the current task enters a taskwait
	//!
	//! \param[in] task The (soon to be blocked) task entering the taskwait
	//! \param[in] taskId Instrumentation id of the task
	//! \param[in] invocationSource The invocation source
	//! \param[in] fromUserCode Whether the taskwait is executed from user code
	//! or runtime code
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

	//! \brief Exit point of the "taskWait" function (TaskWait.cpp)
	//!
	//! Actions:
	//! - HWCounters: Update the runtime's counters as we will begin measuring
	//!   the resumed task's counters
	//! - Monitoring: Notify that the task resumes execution
	//! - Instrument: Notify that the current task exits a taskwait
	//!
	//! \param[in] task The (soon to be resumed) task exiting the taskwait
	//! \param[in] taskId Instrumentation id of the task
	//! \param[in] fromUserCode Whether the taskwait was executed from user code
	//! or runtime code
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

	//! \brief Entry point of the "waitForIf0Task" function (If0Task.hpp)
	//!
	//! Actions:
	//! - HWCounters: Update runtime counters as this thread will be suspended
	//!   and it may migrate
	//! - Instrument: Notify that the current task enters a taskwait and will
	//!   be blocked, and that the current thread will suspend
	//! - Monitoring: Notify that the task becomes blocked (previously runtime,
	//!   as it was creating a task
	//!
	//! \param[in] task The (soon to be blocked) task entering the taskwait
	//! \param[in] taskId Instrumentation id of the task
	//! \param[in] invocationSource The invocation source
	//! \param[in] if0TaskId Instrumentation id of the If0 task preempting the current one
	//! \param[in] thread The (soon to be suspended) current thread
	//! \param[in] cpu The current CPU which the current thread is running on
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

	//! \brief Exit point of the "waitForIf0Task" function (If0Task.hpp)
	//!
	//! Actions:
	//! - HWCounters: None, they are updated in AddTask
	//! - Instrument: Notify that the current task resumes execution and exits
	//!   a taskwait
	//! - Monitoring: Notify that the current task resumes its execution
	//!
	//! \param[in] task The (soon to be resumed) task exiting the taskwait
	//! \param[in] taskId Instrumentation id of the task
	inline void exitWaitForIf0Task(Task *task, Instrument::task_id_t taskId)
	{
		// We don't reset hardware counters as this is done in AddTask after
		// the waitForIf0Task function
		Instrument::taskIsExecuting(taskId, true);
		Instrument::exitTaskWait(taskId, false);

		Monitoring::taskChangedStatus(task, executing_status);
	}

	//! \brief Entry point of the "executeInline" function (If0Task.cpp)
	//!
	//! Actions:
	//! - HWCounters: None, they are updated previously
	//! - Monitoring: Notify that the current task will be blocked since an If0
	//!   task is preempting it
	//! - Instrument: Notify that the current task will be blocked and it is
	//!   entering a taskwait
	//!
	//! \param[in] task The (soon to be blocked) task entering a taskwait
	//! \param[in] taskId Instrumentation id of the task
	//! \param[in] invocationSource The invocation source
	//! \param[in] if0TaskId Instrumentation id of the If0 task preempting the current one
	//! \param[in] hasCode Whether the If0 task that preemtps the current task
	//! has code and, thus, is gonna be executed
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

	//! \brief Exit point of the "executeInline" function (If0Task.hpp)
	//!
	//! Actions:
	//! - HWCounters: None, they are updated previously
	//! - Monitoring: Notify that the current task (previously blocked by the
	//!   If0 task) resumes its execution
	//! - Instrument: Notify that the current task resumes execution and exits
	//!   a taskwait
	//!
	//! \param[in] task The (soon to be resumed) task exiting the taskwait
	//! \param[in] taskId Instrumentation id of the task
	//! \param[in] hasCode Whether the If0 task that preempted the current task
	//! has code and, thus, was executed
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

	//! \brief Entry point of the "spawnFunction" function (SpawnFunction.cpp)
	//!
	//! Actions:
	//! - HWCounters: Update task counters as it will switch status
	//! - Instrument: Notify that the current task is creating another one
	//! - Monitoring: Notify that the current task is creating another one
	//!
	//! \param[in] creator The creator task
	//! \param[in] fromUserCode Whether the task is being created from runtime code
	//! or user code. If the task is created from runtime, there is no creator task
	inline void enterSpawnFunction(Task *creator, bool fromUserCode)
	{
		if (fromUserCode) {
			HardwareCounters::updateTaskCounters(creator);
			Monitoring::taskChangedStatus(creator, paused_status);
		}

		Instrument::enterSpawnFunction(fromUserCode);
	}

	//! \brief Exit point of the "spawnFunction" function (SpawnFunction.cpp)
	//!
	//! Actions:
	//! - HWCounters: Reset runtime counters since the task resumes execution
	//! - Instrument: Notify that the current task resumes execution after
	//!   creating another one and that it exits the creation
	//! - Monitoring: Notify that the current task resumes its execution
	//!
	//! \param[in] creator The (soon to be resumed) creator task
	//! \param[in] fromUserCode Whether the task was created from runtime code or
	//! user code. If the task is created from runtime, there is no creator task
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

	//! \brief Entry point of the "nanos6_user_lock" function (UserMutex.cpp)
	//!
	//! Actions:
	//! - HWCounters: Update task counters as the task is gonna become blocked
	//! - Instrument: Notify that the current task is entering user lock mutex
	//! - Monitoring: Notify that the current task is gonna be blocked
	//!
	//! \param[in] task The (soon to be blocked) task
	inline void enterUserLock(Task *task)
	{
		HardwareCounters::updateTaskCounters(task);
		Monitoring::taskChangedStatus(task, paused_status);
		Instrument::enterUserMutexLock();
	}

	//! \brief Exit point of the "nanos6_user_lock" function (UserMutex.cpp)
	//!
	//! Actions:
	//! - HWCounters: Reset runtime counters as the task will resume execution
	//! - Instrument: Notify that the current task is exiting the user lock mutex
	//! - Monitoring: Notify that the current task is resuming execution
	//!
	//! \param[in] task The (soon to be resumed) task
	inline void exitUserLock(Task *task)
	{
		HardwareCounters::updateRuntimeCounters();
		Instrument::exitUserMutexLock();
		Monitoring::taskChangedStatus(task, executing_status);
	}

	//! \brief Entry point of the "nanos6_user_unlock" function (UserMutex.cpp)
	//!
	//! Actions:
	//! - HWCounters: Update task counters as the task is gonna become blocked
	//! - Instrument: Notify that the current task is entering user unlock mutex
	//! - Monitoring: Notify that the current task is gonna be blocked
	//!
	//! \param[in] task The (soon to be blocked) task
	inline void enterUserUnlock(Task *task)
	{
		HardwareCounters::updateTaskCounters(task);
		Monitoring::taskChangedStatus(task, paused_status);
		Instrument::enterUserMutexUnlock();
	}

	//! \brief Exit point of the "nanos6_user_unlock" function (UserMutex.cpp)
	//!
	//! Actions:
	//! - HWCounters: Reset runtime counters as the task will resume execution
	//! - Instrument: Notify that the current task is exiting the user unlock mutex
	//! - Monitoring: Notify that the current task is resuming execution
	//!
	//! \param[in] task The (soon to be resumed) task
	inline void exitUserUnlock(Task *task)
	{
		HardwareCounters::updateRuntimeCounters();
		Instrument::exitUserMutexUnlock();
		Monitoring::taskChangedStatus(task, executing_status);
	}

	//! \brief Entry point of the "blockCurrentTask" function (BlockingAPI.cpp)
	//!
	//! Actions:
	//! - HWCounters: Read and accumulate task hardware counter since it will be blocked
	//! - Monitoring: Notify that the current task is going to be blocked
	//! - Instrument: Notify that the current task is going to be blocked
	//!
	//! \param[in] task The (soon to be resumed) task
	//! \param[in] taskId The instrumentation id of the task
	//! \param[in] fromUserCode Whether the blocking occured from user code or
	//! was forced from within the runtime
	inline void enterBlockCurrentTask(Task *task, Instrument::task_id_t taskId, bool fromUserCode)
	{
		if (fromUserCode) {
			HardwareCounters::updateTaskCounters(task);
			Monitoring::taskChangedStatus(task, paused_status);
		}
		Instrument::enterBlockCurrentTask(taskId, fromUserCode);
		Instrument::taskIsBlocked(taskId, Instrument::user_requested_blocking_reason);
	}

	//! \brief Exit point of the "blockCurrentTask" function (BlockingAPI.cpp)
	//!
	//! Actions:
	//! - HWCounters: Reset runtime counters since the task resumes execution
	//! - Instrument: Notify that the current task resumes execution after
	//!   being blocked
	//! - Monitoring: Notify that the current task resumes its execution
	//!
	//! \param[in] task The (soon to be resumed) task
	//! \param[in] taskId The instrumentation id of the task
	//! \param[in] fromUserCode Whether the blocking occured from user code or
	//! was forced from within the runtime
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

	//! \brief Entry point of the "unblockCurrentTask" function (BlockingAPI.cpp)
	//!
	//! Actions:
	//! - HWCounters: Read task hardware counter since it will execute runtime code
	//! - Monitoring: Notify that the current task is going to be paused (executing runtime code)
	//! - Instrument: Notify that the current task is going to be executing runtime code
	//!
	//! \param[in] task The task executing runtime code
	//! \param[in] taskId The instrumentation id of the task
	//! \param[in] fromUserCode Whether this happened from user code or was
	//! forced from within the runtime
	inline void enterUnblockCurrentTask(Task *task, Instrument::task_id_t taskId, bool fromUserCode)
	{
		if (fromUserCode) {
			HardwareCounters::updateTaskCounters(task);
			Monitoring::taskChangedStatus(task, paused_status);
		}
		Instrument::enterUnblockTask(taskId, fromUserCode);
	}

	//! \brief Exit point of the "unblockCurrentTask" function (BlockingAPI.cpp)
	//!
	//! Actions:
	//! - HWCounters: Reset runtime counters since the task resumes execution
	//! - Instrument: Notify that the current task resumes execution after
	//!   adding another task to the scheduler
	//! - Monitoring: Notify that the current task resumes its execution
	//!
	//! \param[in] task The (soon to be resumed) task
	//! \param[in] taskId The instrumentation id of the task
	//! \param[in] fromUserCode Whether this happened from user code or was
	//! forced from within the runtime
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

	//! \brief Entry point of the "nanos6_wait_for" function (BlockingAPI.cpp)
	//!
	//! Actions:
	//! - HWCounters: Read task hardware counter since it will be re-added to the scheduler
	//! - Monitoring: No action as it is taken in 'addReadyTask' when being re-added
	//! - Instrument: Notify that the current task is going to be re-added to the scheduler
	//!
	//! \param[in] task The task being re-added to the scheduler
	//! \param[in] taskId The instrumentation id of the task
	inline void enterWaitFor(Task *task, Instrument::task_id_t taskId)
	{
		HardwareCounters::updateTaskCounters(task);
		// We do not notify Monitoring yet, as this will be done in the Scheduler's addReadyTask call
		Instrument::enterWaitFor(taskId);
	}

	//! \brief Exit point of the "nanos6_wait_for" function (BlockingAPI.cpp)
	//!
	//! Actions:
	//! - HWCounters: Reset runtime counters since the task resumes execution
	//! - Instrument: Notify that the current task resumes execution after having
	//!   waited for its deadline
	//! - Monitoring: Notify that the current task resumes its execution
	//!
	//! \param[in] task The (soon to be resumed) task
	//! \param[in] taskId The instrumentation id of the task
	inline void exitWaitFor(Task *task, Instrument::task_id_t taskId)
	{
		HardwareCounters::updateRuntimeCounters();
		Instrument::exitWaitFor(taskId);
		Monitoring::taskChangedStatus(task, executing_status);
	}

	//! \brief Entry point of the "addReadyTasks" function (Scheduler.hpp)
	//!
	//! Actions:
	//! - Monitoring: Notify that all the tasks will become ready
	//! - Instrument: Notify that the thread is entering the addReadyTasks
	//!   function and that all the tasks will become ready
	//!
	//! \param[in] tasks An array of tasks that will become ready
	//! \param[in] numTasks The number of tasks in the previous array
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

	//! \brief Exit point of the "addReadyTasks" function (Scheduler.hpp)
	//!
	//! Actions:
	//! - Instrument: Notify that the thread is exiting the addReadyTasks function
	inline void exitAddReadyTasks()
	{
		Instrument::exitAddReadyTask();
	}

	//! \brief Entry point of the "addReadyTask" function (Scheduler.hpp)
	//!
	//! Actions:
	//! - Instrument: Notify that the thread is entering the addReadyTask function
	//!   and that the task will become ready
	//! - Monitoring: Notify that the task will become ready
	//!
	//! \param[in] task The task being added to the scheduler
	inline void enterAddReadyTask(Task *task)
	{
		Instrument::enterAddReadyTask();
		Instrument::taskIsReady(task->getInstrumentationTaskId());
		Monitoring::taskChangedStatus(task, ready_status);
	}

	//! \brief Exit point of the "addReadyTask" function (Scheduler.hpp)
	//!
	//! Actions:
	//! - Instrument: Notify that the thread is exiting the addReadyTask function
	inline void exitAddReadyTask()
	{
		Instrument::exitAddReadyTask();
	}

	//! \brief Entry point of the "createTask" function (AddTask.cpp)
	//!
	//! Actions:
	//! - HWCounters: Read task hardware counter since it will execute runtime code
	//! - Monitoring: Notify that the current task is going to be in the runtime status
	//!
	//! \param[in] creator The creator task creating another one
	//! \param[in] fromUserCode Whether this happened from user code or was
	//! forced from within the runtime
	inline void enterCreateTask(Task *creator, bool fromUserCode)
	{
		if (fromUserCode) {
			HardwareCounters::updateTaskCounters(creator);
			Monitoring::taskChangedStatus(creator, paused_status);
		}
	}

	//! \brief Entry point of the "submitTask" function (AddTask.cpp)
	//!
	//! Actions:
	//! - HWCounters: Initialize hardware counter structures for the task
	//! - Monitoring: Initialize monitoring structures for the task
	//! - Instrument: Initialize instrument structures for the task and notify
	//!   that the current thread is on the submit phase of the creation of a task
	//!
	//! \param[in] task The created task
	//! \param[in] taskId The instrumentation id of the task
	//! \param[in] fromUserCode Whether this happened from user code or was
	//! forced from within the runtime
	inline void enterSubmitTask(Task *task, Instrument::task_id_t taskId, bool fromUserCode)
	{
		Instrument::enterSubmitTask(fromUserCode);

		HardwareCounters::taskCreated(task);
		Monitoring::taskCreated(task);
		Instrument::createdTask(task, taskId);
	}

	//! \brief Exit point of the "submitTask" function (AddTask.cpp)
	//!
	//! Actions:
	//! - HWCounters: Update runtime counters as the creator will resume execution
	//! - Monitoring: Notify that the creator resumes its execution
	//! - Instrument: Notify that the current thread is no longer creating tasks
	//!
	//! \param[in] creator The creator task
	//! \param[in] taskId The instrumentation id of the created task
	//! \param[in] fromUserCode Whether this happened from user code or was
	//! forced from within the runtime
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
