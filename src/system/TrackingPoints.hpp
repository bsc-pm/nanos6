/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TRACKING_POINTS_HPP
#define TRACKING_POINTS_HPP

#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware-counters/HardwareCounters.hpp"
#include "tasks/Task.hpp"

#include <InstrumentAddTask.hpp>
#include <InstrumentBlockingAPI.hpp>
#include <InstrumentComputePlaceManagement.hpp>
#include <InstrumentScheduler.hpp>
#include <InstrumentTaskExecution.hpp>
#include <InstrumentTaskStatus.hpp>
#include <InstrumentTaskWait.hpp>
#include <InstrumentThreadManagement.hpp>
#include <InstrumentUserMutex.hpp>
#include <Monitoring.hpp>


//! \brief This namespace aggregates common Instrumentation, Monitoring, and
//! HardwareCounter actions in order to simplify the runtime core
namespace TrackingPoints {

	//    COMMON FLOW OF TASKS/THREADS/CPUS    //

	//! \brief Actions to be taken after a task is reinitialized (commonly Taskfors)
	//!
	//! Actions:
	//! - HWCounters: Notify that the current task has reinitialized
	//!
	//! \param[in] task The reinitialized task
	inline void taskReinitialized(Task *task)
	{
		HardwareCounters::taskReinitialized(task);
	}

	//! \brief Actions to be taken when a task switches to pending status
	//!
	//! Actions:
	//! - Instrument: Notify that the task is pending
	//!
	//! \param[in] task The task with unresolved dependencies
	inline void taskIsPending(Task *task)
	{
		Instrument::task_id_t taskId = task->getInstrumentationTaskId();

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
	void taskCompletedUserCode(Task *task);

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
	//! \param[in] thread The thread about to be suspended
	//! \param[in] cpu The CPU the thread is running on
	inline void threadWillSuspend(const WorkerThread *thread, const CPU *cpu)
	{
		assert(cpu != nullptr);
		assert(thread != nullptr);

		Instrument::thread_id_t threadId = thread->getInstrumentationId();
		Instrument::compute_place_id_t cpuId = cpu->getInstrumentationId();
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

	//! \brief Actions to take after a CPU becomes active
	//!
	//! \param[in] cpu The CPU that becomes active
	inline void cpuBecomesActive(const CPU *cpu)
	{
		assert(cpu != nullptr);

		Instrument::resumedComputePlace(cpu->getInstrumentationId());
		Monitoring::cpuBecomesActive(cpu->getIndex());
	}

	//! \brief Actions to take after a CPU becomes idle
	//!
	//! \param[in] cpu The CPU that becomes idle
	//! \param[in] thread The thread currently running in the CPU
	inline void cpuBecomesIdle(const CPU *cpu, const WorkerThread *thread)
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

	//! \brief Entry point of the "taskWait" function (TaskWait.cpp)
	//!
	//! Actions:
	//! - HWCounters: Update (read) the task's counters as it will be blocked
	//! - Monitoring: Notify that the task becomes blocked
	//! - Instrument: Notify that the current task enters a taskwait
	//!
	//! \param[in] task The (soon to be blocked) task entering the taskwait
	//! \param[in] invocationSource The invocation source
	//! \param[in] fromUserCode Whether the taskwait is executed from user code
	//! or runtime code
	inline void enterTaskWait(Task *task, char const *invocationSource, bool fromUserCode)
	{
		assert(task != nullptr);

		Instrument::task_id_t taskId = task->getInstrumentationTaskId();
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
	//! \param[in] fromUserCode Whether the taskwait was executed from user code
	//! or runtime code
	inline void exitTaskWait(Task *task, bool fromUserCode)
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
	//! \param[in] if0Task The if0 task preempting the current one
	//! \param[in] thread The (soon to be suspended) current thread
	//! \param[in] cpu The current CPU which the current thread is running on
	inline void enterWaitForIf0Task(Task *task, const Task *if0Task, const WorkerThread *thread, const CPU *cpu)
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

	//! \brief Exit point of the "waitForIf0Task" function (If0Task.hpp)
	//!
	//! Actions:
	//! - HWCounters: None, they are updated in AddTask
	//! - Instrument: Notify that the current task resumes execution and exits
	//!   a taskwait
	//! - Monitoring: Notify that the current task resumes its execution
	//!
	//! \param[in] task The (soon to be resumed) task exiting the taskwait
	inline void exitWaitForIf0Task(Task *task)
	{
		assert(task != nullptr);

		Instrument::task_id_t taskId = task->getInstrumentationTaskId();

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
	//! \param[in] if0Task The if0 task preempting the current one
	inline void enterExecuteInline(Task *task, const Task *if0Task)
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
	//! \param[in] if0Task The if0 task that was preempting the current one
	inline void exitExecuteInline(Task *task, const Task *if0Task)
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
		// NOTE: Our modules are interested in the transitions between Runtime and Tasks, however,
		// these functions may be called from within the runtime with "fromUserCod" set to true (e.g.
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
	//! \param[in] fromUserCode Whether the blocking occured from user code or
	//! was forced from within the runtime
	inline void enterBlockCurrentTask(Task *task, bool fromUserCode)
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

	//! \brief Exit point of the "blockCurrentTask" function (BlockingAPI.cpp)
	//!
	//! Actions:
	//! - HWCounters: Reset runtime counters since the task resumes execution
	//! - Instrument: Notify that the current task resumes execution after
	//!   being blocked
	//! - Monitoring: Notify that the current task resumes its execution
	//!
	//! \param[in] task The (soon to be resumed) task
	//! \param[in] fromUserCode Whether the blocking occured from user code or
	//! was forced from within the runtime
	inline void exitBlockCurrentTask(Task *task, bool fromUserCode)
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

	//! \brief Entry point of the "unblockTask" function (BlockingAPI.cpp)
	//!
	//! Actions:
	//! - HWCounters: Read task hardware counter since it will execute runtime code
	//! - Monitoring: Notify that the current task is going to be paused (executing runtime code)
	//! - Instrument: Notify that the current task is going to be executing runtime code
	//!
	//! \param[in] task The blocked task
	//! \param[in] currenTask The task executing runtime code
	//! \param[in] fromUserCode Whether this happened from user code or was
	//! forced from within the runtime
	inline void enterUnblockTask(const Task *task, Task *currentTask, bool fromUserCode)
	{
		// NOTE: See the note in "enterSpawnFunction" for more details
		assert(task != nullptr);

		bool taskRuntimeTransition = fromUserCode && (currentTask != nullptr);
		if (taskRuntimeTransition) {
			HardwareCounters::updateTaskCounters(currentTask);
			Monitoring::taskChangedStatus(currentTask, paused_status);
		}

		Instrument::task_id_t taskId = task->getInstrumentationTaskId();
		Instrument::enterUnblockTask(taskId, taskRuntimeTransition);
	}

	//! \brief Exit point of the "unblockTask" function (BlockingAPI.cpp)
	//!
	//! Actions:
	//! - HWCounters: Reset runtime counters since the task resumes execution
	//! - Instrument: Notify that the current task resumes execution after
	//!   adding another task to the scheduler
	//! - Monitoring: Notify that the current task resumes its execution
	//!
	//! \param[in] task The task that was blocked
	//! \param[in] currentTask The (soon to be resumed) task
	//! \param[in] fromUserCode Whether this happened from user code or was
	//! forced from within the runtime
	inline void exitUnblockTask(const Task *task, Task *currentTask, bool fromUserCode)
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

	//! \brief Entry point of the "nanos6_wait_for" function (BlockingAPI.cpp)
	//!
	//! Actions:
	//! - HWCounters: Read task hardware counter since it will be re-added to the scheduler
	//! - Monitoring: No action as it is taken in 'addReadyTask' when being re-added
	//! - Instrument: Notify that the current task is going to be re-added to the scheduler
	//!
	//! \param[in] task The task being re-added to the scheduler
	inline void enterWaitFor(Task *task)
	{
		assert(task != nullptr);

		Instrument::task_id_t taskId = task->getInstrumentationTaskId();
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
	inline void exitWaitFor(Task *task)
	{
		assert(task != nullptr);

		Instrument::task_id_t taskId = task->getInstrumentationTaskId();
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
		assert(task != nullptr);

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
		// NOTE: See the note in "enterSpawnFunction" for more details
		bool taskRuntimeTransition = fromUserCode && (creator != nullptr);
		if (taskRuntimeTransition) {
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
	//! \param[in] creator The creator task
	//! \param[in] task The created task
	//! \param[in] taskId The instrumentation id of the task
	//! \param[in] fromUserCode Whether this happened from user code or was
	//! forced from within the runtime
	inline void enterSubmitTask(Task *creator, Task *task, bool fromUserCode)
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

	//! \brief Exit point of the "submitTask" function (AddTask.cpp)
	//!
	//! Actions:
	//! - HWCounters: Update runtime counters as the creator will resume execution
	//! - Monitoring: Notify that the creator resumes its execution
	//! - Instrument: Notify that the current thread is no longer creating tasks
	//!
	//! \param[in] creator The creator task
	//! \param[in] task The created task
	//! \param[in] fromUserCode Whether this happened from user code or was
	//! forced from within the runtime
	inline void exitSubmitTask(Task *creator, const Task *task, bool fromUserCode)
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

}

#endif // TRACKING_POINTS_HPP
