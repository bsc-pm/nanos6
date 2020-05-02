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
static CTFAPI::CTFEvent *eventThreadResume;
static CTFAPI::CTFEvent *eventThreadSuspend;
static CTFAPI::CTFEvent *eventTaskLabel;
static CTFAPI::CTFEvent *eventTaskExecute;
static CTFAPI::CTFEvent *eventTaskAdd;
static CTFAPI::CTFEvent *eventTaskBlock;
static CTFAPI::CTFEvent *eventTaskEnd;

void Instrument::preinitializeCTFEvents(CTFAPI::CTFMetadata *userMetadata)
{
	// create Events
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
		CTFAPI::CTFContextHWC
	));
	eventTaskLabel = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_label",
		"\t\tinteger { size =  8; align = 8; signed = 0; encoding = UTF8; base = 10; } _label["xstr(ARG_STRING_SIZE)"];\n"
		"\t\tinteger { size = 16; align = 8; signed = 0; encoding = none; base = 10; } _type;\n"
	));
	eventTaskAdd = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_add",
		"\t\tinteger { size = 16; align = 8; signed = 0; encoding = none; base = 10; } _type;\n"
		"\t\tinteger { size = 32; align = 8; signed = 0; encoding = none; base = 10; } _id;\n"
	));
	eventTaskExecute = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_execute",
		"\t\tinteger { size = 32; align = 8; signed = 0; encoding = none; base = 10; } _id;\n",
		CTFAPI::CTFContextHWC
	));
	eventTaskBlock = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_block",
		"\t\tinteger { size = 32; align = 8; signed = 0; encoding = none; base = 10; } _id;\n",
		CTFAPI::CTFContextHWC
	));
	eventTaskEnd = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_end",
		"\t\tinteger { size = 32; align = 8; signed = 0; encoding = none; base = 10; } _id;\n",
		CTFAPI::CTFContextHWC
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

void Instrument::tp_task_label(char *taskLabel, ctf_task_type_id_t taskTypeId)
{
	if (!eventTaskLabel->isEnabled())
		return;

	CTFAPI::tracepoint(eventTaskLabel, taskLabel, taskTypeId);
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
