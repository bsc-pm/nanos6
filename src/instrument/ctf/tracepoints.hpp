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
	void tp_thread_suspend(ctf_thread_id_t tid);
	void tp_thread_resume(ctf_thread_id_t tid);
	void tp_thread_shutdown(ctf_thread_id_t tid);
	void tp_external_thread_create(ctf_thread_id_t tid);
	void tp_external_thread_suspend(ctf_thread_id_t tid);
	void tp_external_thread_resume(ctf_thread_id_t tid);
	void tp_external_thread_shutdown(ctf_thread_id_t tid);

	void tp_worker_enter_busy_wait();
	void tp_worker_exit_busy_wait();

	void tp_task_label(const char *taskLabel, const char *taskSource, ctf_tasktype_id_t taskTypeId);
	void tp_task_execute(ctf_task_id_t taskId);
	void tp_task_create_enter(ctf_tasktype_id_t taskTypeId, ctf_task_id_t taskId);
	void tp_task_create_exit();
	void tp_taskfor_init_enter(ctf_tasktype_id_t taskTypeId, ctf_task_id_t taskId);
	void tp_taskfor_init_exit();
	void tp_task_submit_enter();
	void tp_task_submit_exit();
	void tp_task_block(ctf_task_id_t taskId);
	void tp_task_end(ctf_task_id_t taskId);

	void tp_dependency_register_enter();
	void tp_dependency_register_exit();
	void tp_dependency_unregister_enter();
	void tp_dependency_unregister_exit();

	void tp_scheduler_add_task_enter();
	void tp_scheduler_add_task_exit();
	void tp_scheduler_get_task_enter();
	void tp_scheduler_get_task_exit();

	void tp_polling_service_register(const char *name, ctf_polling_service_id_t id);
	void tp_polling_service_enter(ctf_polling_service_id_t id);
	void tp_polling_service_exit();
}

#endif //NANOS6CTFEVENTS_HPP
