/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6CTFEVENTS_HPP
#define NANOS6CTFEVENTS_HPP

#include "ctfapi/CTFAPI.hpp"
#include "ctfapi/CTFTypes.hpp"
#include "ctfapi/CTFEvent.hpp"
#include "ctfapi/context/CTFContext.hpp"
#include "ctfapi/CTFMetadata.hpp"

namespace Instrument {

	void preinitializeCTFEvents(CTFAPI::CTFMetadata *userMetadata);

	void tp_thread_create(ctf_thread_id_t tid);
	void tp_external_thread_create(ctf_thread_id_t tid);
	void tp_thread_suspend(ctf_thread_id_t tid);
	void tp_thread_resume(ctf_thread_id_t tid);
	void tp_task_label(char *taskLabel, ctf_task_type_id_t taskTypeId);
	void tp_task_execute(ctf_task_id_t taskId);
	void tp_task_add(ctf_task_type_id_t taskTypeId, ctf_task_id_t taskId);
	void tp_task_block(ctf_task_id_t taskId);
	void tp_task_end(ctf_task_id_t taskId);
}

#endif //NANOS6CTFEVENTS_HPP
