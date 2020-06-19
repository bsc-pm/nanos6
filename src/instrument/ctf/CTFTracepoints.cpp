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
static CTFAPI::CTFEvent *eventTaskCreateEnter;
static CTFAPI::CTFEvent *eventTaskCreateExit;
static CTFAPI::CTFEvent *eventTaskforInitEnter;
static CTFAPI::CTFEvent *eventTaskforInitExit;
static CTFAPI::CTFEvent *eventTaskSubmitEnter;
static CTFAPI::CTFEvent *eventTaskSubmitExit;
static CTFAPI::CTFEvent *eventTaskExecute;
static CTFAPI::CTFEvent *eventTaskBlock;
static CTFAPI::CTFEvent *eventTaskEnd;
static CTFAPI::CTFEvent *eventDependencyRegisterEnter;
static CTFAPI::CTFEvent *eventDependencyRegisterExit;
static CTFAPI::CTFEvent *eventDependencyUnregisterEnter;
static CTFAPI::CTFEvent *eventDependencyUnregisterExit;
static CTFAPI::CTFEvent *eventSchedulerAddTaskEnter;
static CTFAPI::CTFEvent *eventSchedulerAddTaskExit;
static CTFAPI::CTFEvent *eventSchedulerGetTaskEnter;
static CTFAPI::CTFEvent *eventSchedulerGetTaskExit;
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
		CTFAPI::CTFContextCPUHWC
	));
	eventThreadShutdown = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:thread_shutdown",
		"\t\tuint16_t _tid;\n",
		CTFAPI::CTFContextCPUHWC
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
	eventTaskCreateEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_create_enter",
		"\t\tuint16_t _type;\n"
		"\t\tuint32_t _id;\n"
	));
	eventTaskCreateExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_create_exit",
		"\t\tuint8_t _dummy;\n"
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
	eventTaskSubmitEnter = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_submit_enter",
		"\t\tuint8_t _dummy;\n"
	));
	eventTaskSubmitExit = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_submit_exit",
		"\t\tuint8_t _dummy;\n"
	));
	eventTaskExecute = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_execute",
		"\t\tuint32_t _id;\n",
		CTFAPI::CTFContextCPUHWC
	));
	eventTaskBlock = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_block",
		"\t\tuint32_t _id;\n",
		CTFAPI::CTFContextTaskHWC
	));
	eventTaskEnd = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_end",
		"\t\tuint32_t _id;\n",
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

void Instrument::tp_task_create_enter(ctf_tasktype_id_t taskTypeId, ctf_task_id_t taskId)
{
	if (!eventTaskCreateEnter->isEnabled())
		return;

	CTFAPI::tracepoint(eventTaskCreateEnter, taskTypeId, taskId);
}

void Instrument::tp_task_create_exit()
{
	if (!eventTaskCreateExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskCreateExit, dummy);
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

void Instrument::tp_task_submit_enter()
{
	if (!eventTaskSubmitEnter->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskSubmitEnter, dummy);
}

void Instrument::tp_task_submit_exit()
{
	if (!eventTaskSubmitExit->isEnabled())
		return;

	char dummy = 0;
	CTFAPI::tracepoint(eventTaskSubmitExit, dummy);
}

void Instrument::tp_task_execute(ctf_task_id_t taskId)
{
	if (!eventTaskExecute->isEnabled())
		return;

	CTFAPI::tracepoint(eventTaskExecute, taskId);
}

void Instrument::tp_task_block(ctf_task_id_t taskId)
{
	if (!eventTaskBlock->isEnabled())
		return;

	CTFAPI::tracepoint(eventTaskBlock, taskId);
}

void Instrument::tp_task_end(ctf_task_id_t taskId)
{
	if (!eventTaskEnd->isEnabled())
		return;

	CTFAPI::tracepoint(eventTaskEnd, taskId);
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
