/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "CTFTracepoints.hpp"

#define xstr(s) str(s)
#define str(s) #s

// TODO can we manage the declaration and calling of a event/tracepoint with a
// macro?

CTFAPI::CTFEvent *__eventCTFFlush;

static CTFAPI::CTFEvent *eventThreadCreate;
static CTFAPI::CTFEvent *eventThreadResume;
static CTFAPI::CTFEvent *eventThreadSuspend;
static CTFAPI::CTFEvent *eventThreadShutdown;
static CTFAPI::CTFEvent *eventExternalThreadCreate;
static CTFAPI::CTFEvent *eventExternalThreadResume;
static CTFAPI::CTFEvent *eventExternalThreadSuspend;
static CTFAPI::CTFEvent *eventExternalThreadShutdown;
static CTFAPI::CTFEvent *eventWorkerEnterBusyWait;
static CTFAPI::CTFEvent *eventWorkerExitBusyWait;
static CTFAPI::CTFEvent *eventTaskLabel;
static CTFAPI::CTFEvent *eventTaskCreateTaskContextEnter;
static CTFAPI::CTFEvent *eventTaskCreateTaskContextExit;
static CTFAPI::CTFEvent *eventTaskCreateOtherContextEnter;
static CTFAPI::CTFEvent *eventTaskCreateOtherContextExit;
static CTFAPI::CTFEvent *eventTaskSubmitTaskContextEnter;
static CTFAPI::CTFEvent *eventTaskSubmitTaskContextExit;
static CTFAPI::CTFEvent *eventTaskSubmitOtherContextEnter;
static CTFAPI::CTFEvent *eventTaskSubmitOtherContextExit;
static CTFAPI::CTFEvent *eventTaskforInitEnter;
static CTFAPI::CTFEvent *eventTaskforInitExit;
static CTFAPI::CTFEvent *eventTaskStart;
static CTFAPI::CTFEvent *eventTaskBlock;
static CTFAPI::CTFEvent *eventTaskUnblock;
static CTFAPI::CTFEvent *eventTaskEnd;
static CTFAPI::CTFEvent *eventDependencyRegisterEnter;
static CTFAPI::CTFEvent *eventDependencyRegisterExit;
static CTFAPI::CTFEvent *eventDependencyUnregisterEnter;
static CTFAPI::CTFEvent *eventDependencyUnregisterExit;
static CTFAPI::CTFEvent *eventSchedulerAddTaskEnter;
static CTFAPI::CTFEvent *eventSchedulerAddTaskExit;
static CTFAPI::CTFEvent *eventSchedulerGetTaskEnter;
static CTFAPI::CTFEvent *eventSchedulerGetTaskExit;
static CTFAPI::CTFEvent *eventTaskwaitTaskContextEnter;
static CTFAPI::CTFEvent *eventTaskwaitTaskContextExit;
static CTFAPI::CTFEvent *eventWaitForTaskContextEnter;
static CTFAPI::CTFEvent *eventWaitForTaskContextExit;
static CTFAPI::CTFEvent *eventBlockingAPIBlockTaskContextEnter;
static CTFAPI::CTFEvent *eventBlockingAPIBlockTaskContextExit;
static CTFAPI::CTFEvent *eventBlockingAPIUnblockTaskContextEnter;
static CTFAPI::CTFEvent *eventBlockingAPIUnblockTaskContextExit;
static CTFAPI::CTFEvent *eventBlockingAPIUnblockOtherContextEnter;
static CTFAPI::CTFEvent *eventBlockingAPIUnblockOtherContextExit;
static CTFAPI::CTFEvent *eventSpawnFunctionTaskContextEnter;
static CTFAPI::CTFEvent *eventSpawnFunctionTaskContextExit;
static CTFAPI::CTFEvent *eventSpawnFunctionOtherContextEnter;
static CTFAPI::CTFEvent *eventSpawnFunctionOtherContextExit;
static CTFAPI::CTFEvent *eventMutexLockTaskContextEnter;
static CTFAPI::CTFEvent *eventMutexLockTaskContextExit;
static CTFAPI::CTFEvent *eventMutexUnlockTaskContextEnter;
static CTFAPI::CTFEvent *eventMutexUnlockTaskContextExit;
static CTFAPI::CTFEvent *eventDebugRegister;
static CTFAPI::CTFEvent *eventDebugEnter;
static CTFAPI::CTFEvent *eventDebugExit;

void Instrument::preinitializeCTFEvents(CTFAPI::CTFMetadata *userMetadata)
{
	// create Events
	__eventCTFFlush = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:ctf_flush",
		"\t\tuint64_t _start;\n"
		"\t\tuint64_t _end;\n"
	));
	eventThreadCreate = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:thread_create",
		"\t\tuint16_t _tid;\n"
	));
	eventThreadResume = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:thread_resume",
		"\t\tuint16_t _tid;\n"
	));
	eventThreadSuspend = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:thread_suspend",
		"\t\tuint16_t _tid;\n",
		CTFAPI::CTFContextRuntimeHWC
	));
	eventThreadShutdown = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:thread_shutdown",
		"\t\tuint16_t _tid;\n",
		CTFAPI::CTFContextRuntimeHWC
	));
	eventExternalThreadCreate = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:external_thread_create",
		"\t\tuint16_t _tid;\n"
	));
	eventExternalThreadResume = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:external_thread_resume",
		"\t\tuint16_t _tid;\n"
	));
	eventExternalThreadSuspend = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:external_thread_suspend",
		"\t\tuint16_t _tid;\n"
	));
	eventExternalThreadShutdown = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:external_thread_shutdown",
		"\t\tuint16_t _tid;\n"
	));
	eventWorkerEnterBusyWait = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:worker_enter_busy_wait",
		"\t\tuint8_t _dummy;\n"
	));
	eventWorkerExitBusyWait = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:worker_exit_busy_wait",
		"\t\tuint8_t _dummy;\n"
	));
	eventTaskLabel = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_label",
		"\t\tstring _label;\n"
		"\t\tstring _source;\n"
		"\t\tuint16_t _type;\n"
	));
	eventTaskCreateTaskContextEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:task_create_enter",
		"\t\tuint16_t _type;\n"
		"\t\tuint32_t _id;\n",
		CTFAPI::CTFContextTaskHWC
	));
	eventTaskCreateTaskContextExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:task_create_exit",
		"\t\tuint8_t _dummy;\n"
	));
	eventTaskCreateOtherContextEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:oc:task_create_enter",
		"\t\tuint16_t _type;\n"
		"\t\tuint32_t _id;\n"
	));
	eventTaskCreateOtherContextExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:oc:task_create_exit",
		"\t\tuint8_t _dummy;\n"
	));
	eventTaskSubmitTaskContextEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:task_submit_enter",
		"\t\tuint8_t _dummy;\n"
	));
	eventTaskSubmitTaskContextExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:task_submit_exit",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextRuntimeHWC
	));
	eventTaskSubmitOtherContextEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:oc:task_submit_enter",
		"\t\tuint8_t _dummy;\n"
	));
	eventTaskSubmitOtherContextExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:oc:task_submit_exit",
		"\t\tuint8_t _dummy;\n"
	));
	eventTaskStart = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_start",
		"\t\tuint32_t _id;\n",
		CTFAPI::CTFContextRuntimeHWC
	));
	eventTaskforInitEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:taskfor_init_enter",
		"\t\tuint16_t _type;\n"
		"\t\tuint32_t _id;\n"
	));
	eventTaskforInitExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:taskfor_init_exit",
		"\t\tuint8_t _dummy;\n"
	));
	eventTaskBlock = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_block",
		"\t\tuint8_t _dummy;\n"
	));
	eventTaskUnblock = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_unblock",
		"\t\tuint8_t _dummy;\n"
	));
	eventTaskEnd = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_end",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextTaskHWC
	));
	eventDependencyRegisterEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:dependency_register_enter",
		"\t\tuint8_t _dummy;\n"
	));
	eventDependencyRegisterExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:dependency_register_exit",
		"\t\tuint8_t _dummy;\n"
	));
	eventDependencyUnregisterEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:dependency_unregister_enter",
		"\t\tuint8_t _dummy;\n"
	));
	eventDependencyUnregisterExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:dependency_unregister_exit",
		"\t\tuint8_t _dummy;\n"
	));
	eventSchedulerAddTaskEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:scheduler_add_task_enter",
		"\t\tuint8_t _dummy;\n"
	));
	eventSchedulerAddTaskExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:scheduler_add_task_exit",
		"\t\tuint8_t _dummy;\n"
	));
	eventSchedulerGetTaskEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:scheduler_get_task_enter",
		"\t\tuint8_t _dummy;\n"
	));
	eventSchedulerGetTaskExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:scheduler_get_task_exit",
		"\t\tuint8_t _dummy;\n"
	));
	eventTaskwaitTaskContextEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:taskwait_enter",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextTaskHWC
	));
	eventTaskwaitTaskContextExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:taskwait_exit",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextRuntimeHWC
	));
	eventWaitForTaskContextEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:waitfor_enter",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextTaskHWC
	));
	eventWaitForTaskContextExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:waitfor_exit",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextRuntimeHWC
	));
	eventBlockingAPIBlockTaskContextEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:blocking_api_block_enter",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextTaskHWC
	));
	eventBlockingAPIBlockTaskContextExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:blocking_api_block_exit",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextRuntimeHWC
	));
	eventBlockingAPIUnblockTaskContextEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:blocking_api_unblock_enter",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextTaskHWC
	));
	eventBlockingAPIUnblockTaskContextExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:blocking_api_unblock_exit",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextRuntimeHWC
	));
	eventBlockingAPIUnblockOtherContextEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:oc:blocking_api_unblock_enter",
		"\t\tuint8_t _dummy;\n"
	));
	eventBlockingAPIUnblockOtherContextExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:oc:blocking_api_unblock_exit",
		"\t\tuint8_t _dummy;\n"
	));
	eventSpawnFunctionTaskContextEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:spawn_function_enter",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextTaskHWC
	));
	eventSpawnFunctionTaskContextExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:spawn_function_exit",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextRuntimeHWC
	));
	eventSpawnFunctionOtherContextEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:oc:spawn_function_enter",
		"\t\tuint8_t _dummy;\n"
	));
	eventSpawnFunctionOtherContextExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:oc:spawn_function_exit",
		"\t\tuint8_t _dummy;\n"
	));
	eventMutexLockTaskContextEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:mutex_lock_enter",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextTaskHWC
	));
	eventMutexLockTaskContextExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:mutex_lock_exit",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextRuntimeHWC
	));
	eventMutexUnlockTaskContextEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:mutex_unlock_enter",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextTaskHWC
	));
	eventMutexUnlockTaskContextExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:tc:mutex_unlock_exit",
		"\t\tuint8_t _dummy;\n",
		CTFAPI::CTFContextRuntimeHWC
	));
	eventDebugRegister = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:debug_register",
		"\t\tstring name;\n"
		"\t\tuint8_t _id;\n"
	));
	eventDebugEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:debug_enter",
		"\t\tuint8_t _id;\n"
	));
	eventDebugExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:debug_exit",
		"\t\tuint8_t _dummy;\n"
	));
}

void Instrument::tp_thread_create(ctf_thread_id_t tid)
{
	if (!eventThreadCreate->isEnabled())
		return;

	CTFAPI::tracepoint(eventThreadCreate, tid);
}

void Instrument::tp_thread_resume(ctf_thread_id_t tid)
{
	if (!eventThreadResume->isEnabled())
		return;

	CTFAPI::tracepoint(eventThreadResume, tid);
}

void Instrument::tp_thread_suspend(ctf_thread_id_t tid)
{
	if (!eventThreadSuspend->isEnabled())
		return;

	CTFAPI::tracepoint(eventThreadSuspend, tid);
}

void Instrument::tp_thread_shutdown(ctf_thread_id_t tid)
{
	if (!eventThreadShutdown->isEnabled())
		return;

	CTFAPI::tracepoint(eventThreadShutdown, tid);
}

void Instrument::tp_external_thread_create(ctf_thread_id_t tid)
{
	if (!eventExternalThreadCreate->isEnabled())
		return;

	CTFAPI::tracepoint(eventExternalThreadCreate, tid);
}

void Instrument::tp_external_thread_resume(ctf_thread_id_t tid)
{
	if (!eventExternalThreadResume->isEnabled())
		return;

	CTFAPI::tracepoint(eventExternalThreadResume, tid);
}

void Instrument::tp_external_thread_suspend(ctf_thread_id_t tid)
{
	if (!eventExternalThreadSuspend->isEnabled())
		return;

	CTFAPI::tracepoint(eventExternalThreadSuspend, tid);
}

void Instrument::tp_external_thread_shutdown(ctf_thread_id_t tid)
{
	if (!eventExternalThreadShutdown->isEnabled())
		return;

	CTFAPI::tracepoint(eventExternalThreadShutdown, tid);
}

void Instrument::tp_worker_enter_busy_wait()
{
	if (!eventWorkerEnterBusyWait->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventWorkerEnterBusyWait, dummy);
}

void Instrument::tp_worker_exit_busy_wait()
{
	if (!eventWorkerExitBusyWait->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventWorkerExitBusyWait, dummy);
}

void Instrument::tp_task_label(const char *taskLabel, const char *taskSource, ctf_tasktype_id_t taskTypeId)
{
	if (!eventTaskLabel->isEnabled())
		return;

	CTFAPI::tracepoint(eventTaskLabel, taskLabel, taskSource, taskTypeId);
}

void Instrument::tp_task_create_tc_enter(ctf_tasktype_id_t taskTypeId, ctf_task_id_t taskId)
{
	if (!eventTaskCreateTaskContextEnter->isEnabled())
		return;

	CTFAPI::tracepoint(eventTaskCreateTaskContextEnter, taskTypeId, taskId);
}

void Instrument::tp_task_create_tc_exit()
{
	if (!eventTaskCreateTaskContextExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskCreateTaskContextExit, dummy);
}

void Instrument::tp_task_create_oc_enter(ctf_tasktype_id_t taskTypeId, ctf_task_id_t taskId)
{
	if (!eventTaskCreateOtherContextEnter->isEnabled())
		return;

	CTFAPI::tracepoint(eventTaskCreateOtherContextEnter, taskTypeId, taskId);
}

void Instrument::tp_task_create_oc_exit()
{
	if (!eventTaskCreateOtherContextExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskCreateOtherContextExit, dummy);
}

void Instrument::tp_taskfor_init_enter(ctf_tasktype_id_t taskTypeId, ctf_task_id_t taskId)
{
	if (!eventTaskforInitEnter->isEnabled())
		return;

	CTFAPI::tracepoint(eventTaskforInitEnter, taskTypeId, taskId);
}

void Instrument::tp_taskfor_init_exit()
{
	if (!eventTaskforInitExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskforInitExit, dummy);
}

void Instrument::tp_task_submit_tc_enter()
{
	if (!eventTaskSubmitTaskContextEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskSubmitTaskContextEnter, dummy);
}

void Instrument::tp_task_submit_tc_exit()
{
	if (!eventTaskSubmitTaskContextExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskSubmitTaskContextExit, dummy);
}

void Instrument::tp_task_submit_oc_enter()
{
	if (!eventTaskSubmitOtherContextEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskSubmitOtherContextEnter, dummy);
}

void Instrument::tp_task_submit_oc_exit()
{
	if (!eventTaskSubmitOtherContextExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskSubmitOtherContextExit, dummy);
}

void Instrument::tp_task_start(ctf_task_id_t taskId)
{
	if (!eventTaskStart->isEnabled())
		return;

	CTFAPI::tracepoint(eventTaskStart, taskId);
}

void Instrument::tp_task_block()
{
	if (!eventTaskBlock->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskBlock, dummy);
}

void Instrument::tp_task_unblock()
{
	if (!eventTaskUnblock->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskUnblock, dummy);
}

void Instrument::tp_task_end()
{
	if (!eventTaskEnd->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskEnd, dummy);
}

void Instrument::tp_dependency_register_enter()
{
	if (!eventDependencyRegisterEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventDependencyRegisterEnter, dummy);
}

void Instrument::tp_dependency_register_exit()
{
	if (!eventDependencyRegisterExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventDependencyRegisterExit, dummy);
}

void Instrument::tp_dependency_unregister_enter()
{
	if (!eventDependencyUnregisterEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventDependencyUnregisterEnter, dummy);
}

void Instrument::tp_dependency_unregister_exit()
{
	if (!eventDependencyUnregisterExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventDependencyUnregisterExit, dummy);
}

void Instrument::tp_scheduler_add_task_enter()
{
	if (!eventSchedulerAddTaskEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventSchedulerAddTaskEnter, dummy);
}

void Instrument::tp_scheduler_add_task_exit()
{
	if (!eventSchedulerAddTaskExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventSchedulerAddTaskExit, dummy);
}

void Instrument::tp_scheduler_get_task_enter()
{
	if (!eventSchedulerGetTaskEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventSchedulerGetTaskEnter, dummy);
}

void Instrument::tp_scheduler_get_task_exit()
{
	if (!eventSchedulerGetTaskExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventSchedulerGetTaskExit, dummy);
}

void Instrument::tp_taskwait_tc_enter()
{
	if (!eventTaskwaitTaskContextEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskwaitTaskContextEnter, dummy);
}

void Instrument::tp_taskwait_tc_exit()
{
	if (!eventTaskwaitTaskContextExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskwaitTaskContextExit, dummy);
}

void Instrument::tp_waitfor_tc_enter()
{
	if (!eventWaitForTaskContextEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventWaitForTaskContextEnter, dummy);
}

void Instrument::tp_waitfor_tc_exit()
{
	if (!eventWaitForTaskContextExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventWaitForTaskContextExit, dummy);
}

void Instrument::tp_blocking_api_block_tc_enter()
{
	if (!eventBlockingAPIBlockTaskContextEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventBlockingAPIBlockTaskContextEnter, dummy);
}

void Instrument::tp_blocking_api_block_tc_exit()
{
	if (!eventBlockingAPIBlockTaskContextExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventBlockingAPIBlockTaskContextExit, dummy);
}

void Instrument::tp_blocking_api_unblock_tc_enter()
{
	if (!eventBlockingAPIUnblockTaskContextEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventBlockingAPIUnblockTaskContextEnter, dummy);
}

void Instrument::tp_blocking_api_unblock_tc_exit()
{
	if (!eventBlockingAPIUnblockTaskContextExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventBlockingAPIUnblockTaskContextExit, dummy);
}

void Instrument::tp_blocking_api_unblock_oc_enter()
{
	if (!eventBlockingAPIUnblockOtherContextEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventBlockingAPIUnblockOtherContextEnter, dummy);
}

void Instrument::tp_blocking_api_unblock_oc_exit()
{
	if (!eventBlockingAPIUnblockOtherContextExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventBlockingAPIUnblockOtherContextExit, dummy);
}

void Instrument::tp_spawn_function_tc_enter()
{
	if (!eventSpawnFunctionTaskContextEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventSpawnFunctionTaskContextEnter, dummy);
}

void Instrument::tp_spawn_function_tc_exit()
{
	if (!eventSpawnFunctionTaskContextExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventSpawnFunctionTaskContextExit, dummy);
}

void Instrument::tp_spawn_function_oc_enter()
{
	if (!eventSpawnFunctionOtherContextEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventSpawnFunctionOtherContextEnter, dummy);
}

void Instrument::tp_spawn_function_oc_exit()
{
	if (!eventSpawnFunctionOtherContextExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventSpawnFunctionOtherContextExit, dummy);
}

void Instrument::tp_mutex_lock_tc_enter()
{
	if (!eventMutexLockTaskContextEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventMutexLockTaskContextEnter, dummy);
}

void Instrument::tp_mutex_lock_tc_exit()
{
	if (!eventMutexLockTaskContextExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventMutexLockTaskContextExit, dummy);
}

void Instrument::tp_mutex_unlock_tc_enter()
{
	if (!eventMutexUnlockTaskContextEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventMutexUnlockTaskContextEnter, dummy);
}

void Instrument::tp_mutex_unlock_tc_exit()
{
	if (!eventMutexUnlockTaskContextExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventMutexUnlockTaskContextExit, dummy);
}

void Instrument::tp_debug_register(const char *name, ctf_debug_id_t id)
{
	if (!eventDebugRegister->isEnabled())
		return;

	CTFAPI::tracepoint(eventDebugRegister, name, id);
}

void Instrument::tp_debug_enter(ctf_debug_id_t id)
{
	if (!eventDebugEnter->isEnabled())
		return;

	CTFAPI::tracepoint(eventDebugEnter, id);
}

void Instrument::tp_debug_exit()
{
	if (!eventDebugExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventDebugExit, dummy);
}
