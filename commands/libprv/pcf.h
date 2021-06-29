/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef LIBPRV_PCF_H
#define LIBPRV_PCF_H

#include "prv.h"

#include <stdio.h>

enum runtime_activity {
	RA_END         = 0,
	RA_RUNTIME     = 1,
	RA_BUSYWAITING = 2,
	RA_TASK        = 3
};

enum runtime_mode {
	RM_DEAD         = 0,
	RM_RUNTIME      = 1,
	RM_TASK         = 3
};

enum runtime_subsystem {
	RS_IDLE = 0,
	RS_RUNTIME,
	RS_BUSY_WAIT,
	RS_TASK,
	RS_DEPENDENCY_REGISTER,
	RS_DEPENDENCY_UNREGISTER,
	RS_SCHEDULER_ADD_TASK,
	RS_SCHEDULER_GET_TASK,
	RS_TASK_CREATE,
	RS_TASK_ARGS_INIT,
	RS_TASK_SUBMIT,
	RS_TASKFOR_INIT,
	RS_TASK_WAIT,
	RS_WAIT_FOR,
	RS_LOCK,
	RS_UNLOCK,
	RS_BLOCKING_API_BLOCK,
	RS_BLOCKING_API_UNBLOCK,
	RS_SPAWN_FUNCTION,
	RS_SCHEDULER_LOCK_ENTER,
	RS_SCHEDULER_LOCK_SERVING
};

enum ev_type {
	EV_TYPE_CTF_FLUSH                 = 6400009,
	EV_TYPE_RUNTIME_CODE              = 6400010,
	EV_TYPE_RUNTIME_BUSYWAITING       = 6400011,
	EV_TYPE_RUNTIME_TASKS             = 6400012,
	EV_TYPE_RUNNING_TASK_LABEL        = 6400013,
	EV_TYPE_RUNNING_TASK_SOURCE       = 6400014,
	EV_TYPE_RUNNING_THREAD_TID        = 6400015,
	EV_TYPE_RUNNING_TASK_ID           = 6400016,
	EV_TYPE_RUNTIME_SUBSYSTEMS        = 6400017,
	EV_TYPE_NUMBER_OF_READY_TASKS     = 6400018,
	EV_TYPE_NUMBER_OF_CREATED_TASKS   = 6400019,
	EV_TYPE_NUMBER_OF_BLOCKED_TASKS   = 6400020,
	EV_TYPE_NUMBER_OF_RUNNING_TASKS   = 6400021,
	EV_TYPE_NUMBER_OF_CREATED_THREADS = 6400022,
	EV_TYPE_NUMBER_OF_RUNNING_THREADS = 6400023,
	EV_TYPE_NUMBER_OF_BLOCKED_THREADS = 6400024,
	EV_TYPE_RUNTIME_MODE              = 6400025,
	EV_TYPE_KERNEL_THREAD_ID          = 6400100,
	EV_TYPE_KERNEL_PREEMPTIONS        = 6400101,
	EV_TYPE_KERNEL_SYSCALLS           = 6400102
};

struct pcf {
	struct task_type *task_types;
};

void
pcf_init(struct pcf *pcf);

int
pcf_write(struct pcf *pcf, FILE *f);

void
pcf_set_task_types(struct pcf *pcf, struct task_type *task_types);

#endif // LIBPRV_PCF_H
