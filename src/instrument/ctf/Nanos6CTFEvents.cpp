/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "Nanos6CTFEvents.hpp"

#define xstr(s) str(s)
#define str(s) #s

// TODO can we manage the declaration and calling of a event/tracepoint with a
// macro?

static CTFAPI::CTFEvent *eventTaskLabel;
static CTFAPI::CTFEvent *eventTaskExecute;
static CTFAPI::CTFEvent *eventTaskAdd;
static CTFAPI::CTFEvent *eventTaskBlock;
static CTFAPI::CTFEvent *eventTaskEnd;
static CTFAPI::CTFEvent *eventCPUResume;
static CTFAPI::CTFEvent *eventCPUIdle;

void Instrument::preinitializeCTFEvents(CTFAPI::CTFMetadata *userMetadata)
{
	// create Events
	eventTaskLabel = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_label",
		"\t\tinteger { size =  8; align = 8; signed = 0; encoding = UTF8; base = 10; } _label["xstr(ARG_STRING_SIZE)"];\n"
		"\t\tinteger { size = 16; align = 8; signed = 0; encoding = none; base = 10; } _type;\n"
	));
	eventTaskExecute = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_execute",
		"\t\tinteger { size = 32; align = 8; signed = 0; encoding = none; base = 10; } _id;\n",
		CTFAPI::CTFContextHWC
	));
	eventTaskAdd = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:task_add",
		"\t\tinteger { size = 16; align = 8; signed = 0; encoding = none; base = 10; } _type;\n"
		"\t\tinteger { size = 32; align = 8; signed = 0; encoding = none; base = 10; } _id;\n"
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
	eventCPUResume = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:cpu_resume",
		"\t\tinteger { size = 16; align = 8; signed = 0; encoding = none; base = 10; } _target;\n"
	));
	eventCPUIdle = userMetadata->addEvent(new CTFAPI::CTFEvent(
		"nanos6:cpu_idle",
		"\t\tinteger { size = 16; align = 8; signed = 0; encoding = none; base = 10; } _target;\n"
	));
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

void Instrument::tp_cpu_idle(uint16_t target)
{
	if (!eventCPUIdle->isEnabled())
		return;

	CTFAPI::tracepoint(eventCPUIdle, target);
}

void Instrument::tp_cpu_resume(uint16_t target)
{
	if (!eventCPUResume->isEnabled())
		return;

	CTFAPI::tracepoint(eventCPUResume, target);
}
