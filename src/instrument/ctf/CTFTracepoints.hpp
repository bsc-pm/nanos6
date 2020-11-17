/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFTRACEPOINTS_HPP
#define CTFTRACEPOINTS_HPP

#include "ctfapi/CTFAPI.hpp"
#include "ctfapi/CTFTypes.hpp"
#include "ctfapi/CTFEvent.hpp"
#include "ctfapi/context/CTFContext.hpp"
#include "ctfapi/CTFUserMetadata.hpp"

namespace Instrument {

	// Management functions

	void preinitializeCTFEvents(CTFAPI::CTFUserMetadata *userMetadata);

	// Internal Nanos6 Tracepoints

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
	void tp_task_start(ctf_task_id_t taskId);
	void tp_taskfor_init_enter(ctf_tasktype_id_t taskTypeId, ctf_task_id_t taskId);
	void tp_taskfor_init_exit();

	void tp_task_block();
	void tp_task_unblock();
	void tp_task_end();

	void tp_dependency_register_enter();
	void tp_dependency_register_exit();
	void tp_dependency_unregister_enter();
	void tp_dependency_unregister_exit();

	void tp_scheduler_add_task_enter();
	void tp_scheduler_add_task_exit();
	void tp_scheduler_get_task_enter();
	void tp_scheduler_get_task_exit();

	void tp_scheduler_lock_client(ctf_timestamp_t acquireTimestamp, ctf_task_id_t taskId);
	void tp_scheduler_lock_server(ctf_timestamp_t acquireTimestamp);
	void tp_scheduler_lock_assign(ctf_task_id_t taskId);
	void tp_scheduler_lock_server_exit();

	// Debug tracepoints

	void tp_debug_register(const char *name, ctf_debug_id_t id);
	void tp_debug_enter(ctf_debug_id_t id);
	void tp_debug_transition(ctf_debug_id_t id);
	void tp_debug_exit();

	// Nanos6 API entry and exit points tracepoints

	void tp_task_create_tc_enter(ctf_tasktype_id_t taskTypeId, ctf_task_id_t taskId);
	void tp_task_create_tc_exit();
	void tp_task_create_oc_enter(ctf_tasktype_id_t taskTypeId, ctf_task_id_t taskId);
	void tp_task_create_oc_exit();

	void tp_task_submit_tc_enter();
	void tp_task_submit_tc_exit();
	void tp_task_submit_oc_enter();
	void tp_task_submit_oc_exit();

	void tp_spawn_function_tc_enter();
	void tp_spawn_function_tc_exit();
	void tp_spawn_function_oc_enter();
	void tp_spawn_function_oc_exit();

	void tp_blocking_api_block_tc_enter();
	void tp_blocking_api_block_tc_exit();

	void tp_blocking_api_unblock_tc_enter();
	void tp_blocking_api_unblock_tc_exit();
	void tp_blocking_api_unblock_oc_enter();
	void tp_blocking_api_unblock_oc_exit();

	void tp_taskwait_tc_enter();
	void tp_taskwait_tc_exit();

	void tp_waitfor_tc_enter();
	void tp_waitfor_tc_exit();

	void tp_mutex_lock_tc_enter();
	void tp_mutex_lock_tc_exit();

	void tp_mutex_unlock_tc_enter();
	void tp_mutex_unlock_tc_exit();
}

#endif //CTFTRACEPOINTS_HPP
