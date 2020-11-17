/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef TRACKING_POINTS_HPP
#define TRACKING_POINTS_HPP

#include <cstddef>


class CPU;
class Task;
class WorkerThread;

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
	void taskReinitialized(Task *task);

	//! \brief Actions to be taken when a task switches to pending status
	//!
	//! Actions:
	//! - Instrument: Notify that the task is pending
	//!
	//! \param[in] task The task with unresolved dependencies
	void taskIsPending(Task *task);

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
	void taskFinished(Task *task);

	//! \brief Actions to be taken after a worker thread initializes
	//!
	//! \param[in] thread The worker thread
	//! \param[in] cpu The current CPU the thread is running on
	void threadInitialized(const WorkerThread *thread, const CPU *cpu);

	//! \brief Common actions within the namespace for when a thread is suspending
	//!
	//! Actions:
	//! - HWCounters: Update runtime counters as this thread will be suspended
	//!   and it may migrate from CPU
	//! - Instrument: Notify that the current thread will suspend
	//!
	//! \param[in] thread The thread about to be suspended
	//! \param[in] cpu The CPU the thread is running on
	void threadWillSuspend(const WorkerThread *thread, const CPU *cpu);

	//! \brief Actions to take after a thread is shutting down
	//!
	//! Actions:
	//! - HWCounters: Update runtime counters for the thread and shutdown counter reading
	//! - Instrument: Notify that the thread is shutting down
	//!
	void threadWillShutdown();

	//! \brief Actions to take after a CPU becomes active
	//!
	//! \param[in] cpu The CPU that becomes active
	void cpuBecomesActive(const CPU *cpu);

	//! \brief Actions to take after a CPU becomes idle
	//!
	//! \param[in] cpu The CPU that becomes idle
	//! \param[in] thread The thread currently running in the CPU
	void cpuBecomesIdle(const CPU *cpu, const WorkerThread *thread);


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
	void enterTaskWait(Task *task, char const *invocationSource, bool fromUserCode);

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
	void exitTaskWait(Task *task, bool fromUserCode);

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
	void enterWaitForIf0Task(Task *task, const Task *if0Task, const WorkerThread *thread, const CPU *cpu);

	//! \brief Exit point of the "waitForIf0Task" function (If0Task.hpp)
	//!
	//! Actions:
	//! - HWCounters: None, they are updated in AddTask
	//! - Instrument: Notify that the current task resumes execution and exits
	//!   a taskwait
	//! - Monitoring: Notify that the current task resumes its execution
	//!
	//! \param[in] task The (soon to be resumed) task exiting the taskwait
	void exitWaitForIf0Task(Task *task);

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
	void enterExecuteInline(Task *task, const Task *if0Task);

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
	void exitExecuteInline(Task *task, const Task *if0Task);

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
	void enterSpawnFunction(Task *creator, bool fromUserCode);

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
	void exitSpawnFunction(Task *creator, bool fromUserCode);

	//! \brief Entry point of the "nanos6_user_lock" function (UserMutex.cpp)
	//!
	//! Actions:
	//! - HWCounters: Update task counters as the task is gonna become blocked
	//! - Instrument: Notify that the current task is entering user lock mutex
	//! - Monitoring: Notify that the current task is gonna be blocked
	//!
	//! \param[in] task The (soon to be blocked) task
	void enterUserLock(Task *task);

	//! \brief Exit point of the "nanos6_user_lock" function (UserMutex.cpp)
	//!
	//! Actions:
	//! - HWCounters: Reset runtime counters as the task will resume execution
	//! - Instrument: Notify that the current task is exiting the user lock mutex
	//! - Monitoring: Notify that the current task is resuming execution
	//!
	//! \param[in] task The (soon to be resumed) task
	void exitUserLock(Task *task);

	//! \brief Entry point of the "nanos6_user_unlock" function (UserMutex.cpp)
	//!
	//! Actions:
	//! - HWCounters: Update task counters as the task is gonna become blocked
	//! - Instrument: Notify that the current task is entering user unlock mutex
	//! - Monitoring: Notify that the current task is gonna be blocked
	//!
	//! \param[in] task The (soon to be blocked) task
	void enterUserUnlock(Task *task);

	//! \brief Exit point of the "nanos6_user_unlock" function (UserMutex.cpp)
	//!
	//! Actions:
	//! - HWCounters: Reset runtime counters as the task will resume execution
	//! - Instrument: Notify that the current task is exiting the user unlock mutex
	//! - Monitoring: Notify that the current task is resuming execution
	//!
	//! \param[in] task The (soon to be resumed) task
	void exitUserUnlock(Task *task);

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
	void enterBlockCurrentTask(Task *task, bool fromUserCode);

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
	void exitBlockCurrentTask(Task *task, bool fromUserCode);

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
	void enterUnblockTask(const Task *task, Task *currentTask, bool fromUserCode);

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
	void exitUnblockTask(const Task *task, Task *currentTask, bool fromUserCode);

	//! \brief Entry point of the "nanos6_wait_for" function (BlockingAPI.cpp)
	//!
	//! Actions:
	//! - HWCounters: Read task hardware counter since it will be re-added to the scheduler
	//! - Monitoring: No action as it is taken in 'addReadyTask' when being re-added
	//! - Instrument: Notify that the current task is going to be re-added to the scheduler
	//!
	//! \param[in] task The task being re-added to the scheduler
	void enterWaitFor(Task *task);

	//! \brief Exit point of the "nanos6_wait_for" function (BlockingAPI.cpp)
	//!
	//! Actions:
	//! - HWCounters: Reset runtime counters since the task resumes execution
	//! - Instrument: Notify that the current task resumes execution after having
	//!   waited for its deadline
	//! - Monitoring: Notify that the current task resumes its execution
	//!
	//! \param[in] task The (soon to be resumed) task
	void exitWaitFor(Task *task);

	//! \brief Entry point of the "addReadyTasks" function (Scheduler.hpp)
	//!
	//! Actions:
	//! - Monitoring: Notify that all the tasks will become ready
	//! - Instrument: Notify that the thread is entering the addReadyTasks
	//!   function and that all the tasks will become ready
	//!
	//! \param[in] tasks An array of tasks that will become ready
	//! \param[in] numTasks The number of tasks in the previous array
	void enterAddReadyTasks(Task *tasks[], const size_t numTasks);

	//! \brief Exit point of the "addReadyTasks" function (Scheduler.hpp)
	//!
	//! Actions:
	//! - Instrument: Notify that the thread is exiting the addReadyTasks function
	void exitAddReadyTasks();

	//! \brief Entry point of the "addReadyTask" function (Scheduler.hpp)
	//!
	//! Actions:
	//! - Instrument: Notify that the thread is entering the addReadyTask function
	//!   and that the task will become ready
	//! - Monitoring: Notify that the task will become ready
	//!
	//! \param[in] task The task being added to the scheduler
	void enterAddReadyTask(Task *task);

	//! \brief Exit point of the "addReadyTask" function (Scheduler.hpp)
	//!
	//! Actions:
	//! - Instrument: Notify that the thread is exiting the addReadyTask function
	void exitAddReadyTask();

	//! \brief Entry point of the "createTask" function (AddTask.cpp)
	//!
	//! Actions:
	//! - HWCounters: Read task hardware counter since it will execute runtime code
	//! - Monitoring: Notify that the current task is going to be in the runtime status
	//!
	//! \param[in] creator The creator task creating another one
	//! \param[in] fromUserCode Whether this happened from user code or was
	//! forced from within the runtime
	void enterCreateTask(Task *creator, bool fromUserCode);

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
	void enterSubmitTask(Task *creator, Task *task, bool fromUserCode);

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
	void exitSubmitTask(Task *creator, const Task *task, bool fromUserCode);

}

#endif // TRACKING_POINTS_HPP
