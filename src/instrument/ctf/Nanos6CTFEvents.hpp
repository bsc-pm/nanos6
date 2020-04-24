/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6CTFEVENTS_HPP
#define NANOS6CTFEVENTS_HPP

#include "ctfapi/CTFAPI.hpp"
#include "ctfapi/CTFEvent.hpp"
#include "ctfapi/CTFContext.hpp"
#include "ctfapi/CTFMetadata.hpp"

namespace Instrument {

	void preinitializeCTFEvents(CTFAPI::CTFMetadata *userMetadata);

	void tp_task_label(char *taskLabel, ctf_task_type_id_t taskTypeId);
	void tp_task_execute(ctf_task_id_t taskId);
	void tp_task_add(ctf_task_type_id_t taskTypeId, ctf_task_id_t taskId);
	void tp_task_block(ctf_task_id_t taskId);
	void tp_task_end(ctf_task_id_t taskId);
	void tp_cpu_idle(uint16_t target);
	void tp_cpu_resume(uint16_t target);

}

#endif //NANOS6CTFEVENTS_HPP
