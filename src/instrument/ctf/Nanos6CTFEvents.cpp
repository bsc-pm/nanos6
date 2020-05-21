/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "Nanos6CTFEvents.hpp"

#define xstr(s) str(s)
#define str(s) #s

// TODO can we manage the declaration and calling of a event/tracepoint with a
// macro?

static CTFAPI::CTFEvent *eventThreadCreate;
static CTFAPI::CTFEvent *eventExternalThreadCreate;
static CTFAPI::CTFEvent *eventThreadResume;
static CTFAPI::CTFEvent *eventThreadSuspend;
static CTFAPI::CTFEvent *eventThreadShutdown;
static CTFAPI::CTFEvent *eventWorkerEnterBusyWait;
static CTFAPI::CTFEvent *eventWorkerExitBusyWait;
static CTFAPI::CTFEvent *eventTaskLabel;
static CTFAPI::CTFEvent *eventTaskExecute;
static CTFAPI::CTFEvent *eventTaskAdd;
static CTFAPI::CTFEvent *eventTaskBlock;
static CTFAPI::CTFEvent *eventTaskEnd;
static CTFAPI::CTFEvent *eventDependencyRegisterEnter;
static CTFAPI::CTFEvent *eventDependencyRegisterExit;
static CTFAPI::CTFEvent *eventDependencyUnregisterEnter;
static CTFAPI::CTFEvent *eventDependencyUnregisterExit;

void Instrument::preinitializeCTFEvents(CTFAPI::CTFMetadata *userMetadata)
{
	// create Events
	eventThreadCreate = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:thread_create",
		"\t\tuint16_t _tid;\n"
	));
	eventExternalThreadCreate = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:external_thread_create",
		"\t\tuint16_t _tid;\n"
	));
	eventThreadResume = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:thread_resume",
		"\t\tuint16_t _tid;\n"
	));
	eventThreadSuspend = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:thread_suspend",
		"\t\tuint16_t _tid;\n",
		CTFAPI::CTFContextHWC
	));
	eventThreadShutdown = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:thread_shutdown",
		"\t\tuint16_t _tid;\n",
		CTFAPI::CTFContextHWC
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
	eventTaskAdd = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_add",
		"\t\tuint16_t _type;\n"
		"\t\tuint32_t _id;\n"
	));
	eventTaskExecute = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_execute",
		"\t\tuint32_t _id;\n",
		CTFAPI::CTFContextHWC
	));
	eventTaskBlock = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_block",
		"\t\tuint32_t _id;\n",
		CTFAPI::CTFContextHWC
	));
	eventTaskEnd = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_end",
		"\t\tuint32_t _id;\n",
		CTFAPI::CTFContextHWC
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
}

void Instrument::tp_thread_create(ctf_thread_id_t tid)
{
	if (!eventThreadCreate->isEnabled())
		return;

	CTFAPI::tracepoint(eventThreadCreate, tid);
}

void Instrument::tp_external_thread_create(ctf_thread_id_t tid)
{
	if (!eventExternalThreadCreate->isEnabled())
		return;

	CTFAPI::tracepoint(eventExternalThreadCreate, tid);
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

void Instrument::tp_task_label(const char *taskLabel, const char *taskSource, ctf_task_type_id_t taskTypeId)
{
	if (!eventTaskLabel->isEnabled())
		return;

	CTFAPI::tracepoint(eventTaskLabel, taskLabel, taskSource, taskTypeId);
}

void Instrument::tp_task_execute(ctf_task_id_t taskId)
{
	if (!eventTaskExecute->isEnabled())
		return;

	CTFAPI::tracepoint(eventTaskExecute, taskId);
}

void Instrument::tp_task_add(ctf_task_type_id_t taskTypeId, ctf_task_id_t taskId)
{
	if (!eventTaskAdd->isEnabled())
		return;

	CTFAPI::tracepoint(eventTaskAdd, taskTypeId, taskId);
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
