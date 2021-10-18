/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <babeltrace2/babeltrace.h>
#include "uthash.h"
#include <time.h>
#include <linux/limits.h>

#include "prv.h"
#include "pcf.h"
#include "hwc.h"

/* Manually extracted */
#define MAX_CPUS 1024
#define MAX_ID 128
#define MAX_HWC 64
#define MAX_FILTERS 16

#define TRACE_NAME "trace"

#define PRV_HEADER_FMT \
	"#Paraver (09/09/41 at 03:14):%020d_ns:0:1:1(%020d:1)\n"

//#define ENABLE_DEBUG

#ifdef ENABLE_DEBUG
#define dbg(...) fprintf(stderr, __VA_ARGS__);
#else
#define dbg(...)
#endif

#define err(...) fprintf(stderr, __VA_ARGS__);

/* Report status every 100 ms */
#define REPORT_TIME 80e-3

/* If defined, multiple events happening at the same time are printed in
 * the same line, saving space, but increasing the complexity to be
 * processed as a stream with common text tools. Otherwise, if not
 * defined, only one event per line is printed. */
#define JOIN_EVENTS_IN_ONE_LINE

enum stream_type {
	STREAM_BOUNDED = 1,
	STREAM_UNBOUNDED = 2
};

enum class_id {
/*  1 */  CLASS_ID_CTF_FLUSH = 1,
/*  2 */  CLASS_ID_THREAD_CREATE,
/*  3 */  CLASS_ID_THREAD_RESUME,
/*  4 */  CLASS_ID_THREAD_SUSPEND,
/*  5 */  CLASS_ID_THREAD_SHUTDOWN,
/*  6 */  CLASS_ID_EXTERNAL_THREAD_CREATE,
/*  7 */  CLASS_ID_EXTERNAL_THREAD_RESUME,
/*  8 */  CLASS_ID_EXTERNAL_THREAD_SUSPEND,
/*  9 */  CLASS_ID_EXTERNAL_THREAD_SHUTDOWN,
/* 10 */  CLASS_ID_WORKER_ENTER_BUSY_WAIT,
/* 11 */  CLASS_ID_WORKER_EXIT_BUSY_WAIT,
/* 12 */  CLASS_ID_TASK_LABEL,
/* 13 */  CLASS_ID_TC_TASK_CREATE_ENTER,
/* 14 */  CLASS_ID_TC_TASK_CREATE_EXIT,
/* 15 */  CLASS_ID_OC_TASK_CREATE_ENTER,
/* 16 */  CLASS_ID_OC_TASK_CREATE_EXIT,
/* 17 */  CLASS_ID_TC_TASK_SUBMIT_ENTER,
/* 18 */  CLASS_ID_TC_TASK_SUBMIT_EXIT,
/* 19 */  CLASS_ID_OC_TASK_SUBMIT_ENTER,
/* 20 */  CLASS_ID_OC_TASK_SUBMIT_EXIT,
/* 21 */  CLASS_ID_TASK_START,
/* 22 */  CLASS_ID_TASKFOR_INIT_ENTER,
/* 23 */  CLASS_ID_TASKFOR_INIT_EXIT,
/* 24 */  CLASS_ID_TASK_BLOCK,
/* 25 */  CLASS_ID_TASK_UNBLOCK,
/* 26 */  CLASS_ID_TASK_END,
/* 27 */  CLASS_ID_DEPENDENCY_REGISTER_ENTER,
/* 28 */  CLASS_ID_DEPENDENCY_REGISTER_EXIT,
/* 29 */  CLASS_ID_DEPENDENCY_UNREGISTER_ENTER,
/* 30 */  CLASS_ID_DEPENDENCY_UNREGISTER_EXIT,
/* 31 */  CLASS_ID_SCHEDULER_ADD_TASK_ENTER,
/* 32 */  CLASS_ID_SCHEDULER_ADD_TASK_EXIT,
/* 33 */  CLASS_ID_SCHEDULER_GET_TASK_ENTER,
/* 34 */  CLASS_ID_SCHEDULER_GET_TASK_EXIT,
/* 35 */  CLASS_ID_TC_TASKWAIT_ENTER,
/* 36 */  CLASS_ID_TC_TASKWAIT_EXIT,
/* 37 */  CLASS_ID_TC_WAITFOR_ENTER,
/* 38 */  CLASS_ID_TC_WAITFOR_EXIT,
/* 39 */  CLASS_ID_TC_BLOCKING_API_BLOCK_ENTER,
/* 40 */  CLASS_ID_TC_BLOCKING_API_BLOCK_EXIT,
/* 41 */  CLASS_ID_TC_BLOCKING_API_UNBLOCK_ENTER,
/* 42 */  CLASS_ID_TC_BLOCKING_API_UNBLOCK_EXIT,
/* 43 */  CLASS_ID_OC_BLOCKING_API_UNBLOCK_ENTER,
/* 44 */  CLASS_ID_OC_BLOCKING_API_UNBLOCK_EXIT,
/* 45 */  CLASS_ID_TC_SPAWN_FUNCTION_ENTER,
/* 46 */  CLASS_ID_TC_SPAWN_FUNCTION_EXIT,
/* 47 */  CLASS_ID_OC_SPAWN_FUNCTION_ENTER,
/* 48 */  CLASS_ID_OC_SPAWN_FUNCTION_EXIT,
/* 49 */  CLASS_ID_TC_MUTEX_LOCK_ENTER,
/* 50 */  CLASS_ID_TC_MUTEX_LOCK_EXIT,
/* 51 */  CLASS_ID_TC_MUTEX_UNLOCK_ENTER,
/* 52 */  CLASS_ID_TC_MUTEX_UNLOCK_EXIT,

/* 57 */  CLASS_ID_SCHEDULER_LOCK_SERVER = 57,
/* 58 */  CLASS_ID_SCHEDULER_LOCK_CLIENT,
/* 59 */  CLASS_ID_SCHEDULER_LOCK_ASSIGN,
/* 60 */  CLASS_ID_SCHEDULER_LOCK_SERVER_EXIT,

/* Fake events that we use internally */
/* 666*/  CLASS_ID_SCHEDULER_LOCK_ACQUIRED = 666
};

enum task_st {
	TASK_ST_UNINIT,
	TASK_ST_RUNNING,
	TASK_ST_BLOCKED
};

struct task {
	uint64_t id; /* Task ID */
	uint64_t type;
	enum task_st state;
	UT_hash_handle hh; /* makes this structure hashable */
};

enum thread_st {
	THREAD_ST_UNKNOWN,
	THREAD_ST_CREATED,
	THREAD_ST_RESUMED,
	THREAD_ST_SUSPENDED
};

#define MAX_EV_STACK 256

struct thread {
	uint64_t tid; /* Thread ID */
	int cpu; /* On which CPU is the thread running on */
	int external; /* Wether the thread is external */
	int state; /* Thread state */
	struct task *task; /* Task running in the thread or NULL */
	int busy_wait; /* 1 if is busy waiting */
	int color; /* Paraver color */
	int ev_stack[MAX_EV_STACK];
	int n_stack; /* Current number of events in the stack */
	UT_hash_handle hh; /* makes this structure hashable */
};

#define MAX_HOOKS 10


struct conv;
typedef void (*hook_func_t)(struct conv *conv,
		const struct bt_event *ev, int class_id);

struct hook_entry {
	int class_id;
	hook_func_t func;
};

struct prv_event {
	long long type;
	long long value;
};

#define MAX_ACC_EVENTS 100

struct cpu {
	int pcpu; /* Physical CPU number */
	struct thread *thread; /* Thread running or NULL */
};

/* Sink component's private data */
struct conv {
	/* Upstream message iterator (owned by this) */
	bt_message_iterator *message_iterator;

	/* Current event message index */
	uint64_t index;

	/* Dummy counter */
	uint64_t dummy;

	/* Hash table for the threads */
	struct thread *threads;

	/* Hash table for the tasks */
	struct task *tasks;

	/* Hash table for the task types */
	struct task_type *task_types;

	/* Hook table */
	hook_func_t hook_table[MAX_ID][MAX_HOOKS];

	/* Event to subsystem table */
	int ss_table[MAX_ID];

	/* Accumulate events and print them in one line */
	struct prv_event acc_ev[MAX_ACC_EVENTS];
	int n_acc_ev;

	int last_thread_color;

	/* The state of each CPU */
	struct cpu cpus[MAX_CPUS];

	/* Physical CPU to CPU index */
	int pcpu_index[MAX_CPUS];

	/* Largest physical CPU */
	int max_pcpu;

	/* Index in conv->cpus of the current event CPU (or virtual CPU) */
	int curr_cpu;

	/* Number of virtual CPUs in use */
	int nvcpus;

	/* Number of physical CPUs */
	int ncpus;

	uint64_t ev_processed;
	uint64_t ev_ignored;
	uint64_t ev_emitted;
	uint64_t ev_last;
	double ev_progress;

	/* Time of the last report */
	double last_reported;

	/* Bytes written in the last report */
	size_t last_written;

	/* The current time */
	double tic;
	double t0;

	/* Where to place output files: must exist */
	const char *output_dir;
	FILE *prv;
	FILE *pcf_file;
	char row_file_path[PATH_MAX];
	struct pcf pcf;

	/* Split events one per line */
	int split_events;

	/* Be quiet */
	int quiet;

	/* HWC: 1 = enabled, 0 = disabled, -1 = not checked */
	int hwc_enabled;

	/* Number of HW counters */
	int nhwc;

	/* HWC PRV event ids */
	int hwc_table[MAX_HWC];

	/* The clock offset corrects the value of the timestamp field in
	 * each event, so that the obtained time value is synced between
	 * ranks. This value must be *added* to each event timestamp to
	 * obtain the resulting time (it may be negative).
	 *
	 * The babelttrace2 clock already applies this offset, but some
	 * events (sched lock) contain timestamps which need to be
	 * corrected.
	 *
	 * TODO: Remove this offset and use a normal CTF event. */
	int64_t clock_offset_ns;

	/* The start and end time values */
	int64_t clock_start_ns;
	int64_t clock_end_ns;

	/* Number of external threads (extra rows) */
	int64_t external_threads;

	/* Number of event filters */
	int nfilters;

	/* Array of events that are filtered */
	long filters[MAX_FILTERS];
};

/* When a event needs to stack a subsystem in the thread stack, we
 * compute the subsystem from the event ID using this list. The ss_table
 * is build for quick access. */
int ss_list[][2] = {
  /* Event					Subsystem */
  { CLASS_ID_THREAD_CREATE,			RS_RUNTIME },
  { CLASS_ID_THREAD_SUSPEND,			RS_IDLE },
  { CLASS_ID_THREAD_SHUTDOWN,			RS_IDLE },
  { CLASS_ID_THREAD_RESUME,			-1 },

  /* FIXME: We need to distinguish between the Leader thread and
   * External threads. Otherwise we paint as Idle after the first resume
   * event */
  { CLASS_ID_EXTERNAL_THREAD_CREATE,		RS_IDLE },
  { CLASS_ID_EXTERNAL_THREAD_SUSPEND,		RS_IDLE },
  { CLASS_ID_EXTERNAL_THREAD_SHUTDOWN,		RS_IDLE },
  { CLASS_ID_EXTERNAL_THREAD_RESUME,		-1 },

  { CLASS_ID_TASK_END,				RS_TASK },
  { CLASS_ID_TASK_START,			RS_TASK },

  { CLASS_ID_TC_TASKWAIT_ENTER,			RS_TASK_WAIT },
  { CLASS_ID_TC_TASKWAIT_EXIT,			RS_TASK_WAIT },

  { CLASS_ID_TC_WAITFOR_ENTER,			RS_WAIT_FOR },
  { CLASS_ID_TC_WAITFOR_EXIT,			RS_WAIT_FOR },

  { CLASS_ID_TC_MUTEX_LOCK_ENTER,		RS_LOCK },
  { CLASS_ID_TC_MUTEX_LOCK_EXIT,		RS_LOCK },
  { CLASS_ID_TC_MUTEX_UNLOCK_ENTER,		RS_UNLOCK },
  { CLASS_ID_TC_MUTEX_UNLOCK_EXIT,		RS_UNLOCK },

  { CLASS_ID_TC_BLOCKING_API_BLOCK_ENTER,	RS_BLOCKING_API_BLOCK },
  { CLASS_ID_TC_BLOCKING_API_BLOCK_EXIT,	RS_BLOCKING_API_BLOCK },
  { CLASS_ID_TC_BLOCKING_API_UNBLOCK_ENTER,	RS_BLOCKING_API_UNBLOCK },
  { CLASS_ID_TC_BLOCKING_API_UNBLOCK_EXIT,	RS_BLOCKING_API_UNBLOCK },
  { CLASS_ID_OC_BLOCKING_API_UNBLOCK_ENTER,	RS_BLOCKING_API_UNBLOCK },
  { CLASS_ID_OC_BLOCKING_API_UNBLOCK_EXIT,	RS_BLOCKING_API_UNBLOCK },

  { CLASS_ID_TC_SPAWN_FUNCTION_ENTER,		RS_SPAWN_FUNCTION },
  { CLASS_ID_TC_SPAWN_FUNCTION_EXIT,		RS_SPAWN_FUNCTION },
  { CLASS_ID_OC_SPAWN_FUNCTION_ENTER,		RS_SPAWN_FUNCTION },
  { CLASS_ID_OC_SPAWN_FUNCTION_EXIT,		RS_SPAWN_FUNCTION },

  { CLASS_ID_WORKER_ENTER_BUSY_WAIT,		RS_BUSY_WAIT },
  { CLASS_ID_WORKER_EXIT_BUSY_WAIT,		RS_BUSY_WAIT },

  { CLASS_ID_DEPENDENCY_REGISTER_ENTER,		RS_DEPENDENCY_REGISTER },
  { CLASS_ID_DEPENDENCY_REGISTER_EXIT,		RS_DEPENDENCY_REGISTER },
  { CLASS_ID_DEPENDENCY_UNREGISTER_ENTER,	RS_DEPENDENCY_UNREGISTER },
  { CLASS_ID_DEPENDENCY_UNREGISTER_EXIT,	RS_DEPENDENCY_UNREGISTER },

  { CLASS_ID_SCHEDULER_ADD_TASK_ENTER,		RS_SCHEDULER_ADD_TASK },
  { CLASS_ID_SCHEDULER_ADD_TASK_EXIT,		RS_SCHEDULER_ADD_TASK },
  { CLASS_ID_SCHEDULER_GET_TASK_ENTER,		RS_SCHEDULER_GET_TASK },
  { CLASS_ID_SCHEDULER_GET_TASK_EXIT,		RS_SCHEDULER_GET_TASK },

  { CLASS_ID_TC_TASK_CREATE_ENTER,		RS_TASK_CREATE },
  { CLASS_ID_TC_TASK_CREATE_EXIT,		RS_TASK_CREATE },
  { CLASS_ID_OC_TASK_CREATE_ENTER,		RS_TASK_CREATE },
  { CLASS_ID_OC_TASK_CREATE_EXIT,		RS_TASK_CREATE },

  { CLASS_ID_TC_TASK_SUBMIT_ENTER,		RS_TASK_SUBMIT },
  { CLASS_ID_TC_TASK_SUBMIT_EXIT,		RS_TASK_SUBMIT },
  { CLASS_ID_OC_TASK_SUBMIT_ENTER,		RS_TASK_SUBMIT },
  { CLASS_ID_OC_TASK_SUBMIT_EXIT,		RS_TASK_SUBMIT },

  { CLASS_ID_TASKFOR_INIT_ENTER,		RS_TASKFOR_INIT },
  { CLASS_ID_TASKFOR_INIT_EXIT,			RS_TASKFOR_INIT },

  { CLASS_ID_SCHEDULER_LOCK_CLIENT,		RS_SCHEDULER_LOCK_ENTER },
  { CLASS_ID_SCHEDULER_LOCK_SERVER,		RS_SCHEDULER_LOCK_SERVING },
  { CLASS_ID_SCHEDULER_LOCK_ASSIGN,		-1 },
  { CLASS_ID_SCHEDULER_LOCK_SERVER_EXIT,	-1 },

  /* End marker */
  { -1,						-1},
};

#define HOOK_DEF(name) \
	void name(struct conv *conv, const struct bt_event *ev, int class_id)

HOOK_DEF(hook_thread_create);
HOOK_DEF(hook_ext_thread_create);
HOOK_DEF(hook_thread_resume);
HOOK_DEF(hook_thread_suspend);
HOOK_DEF(hook_task_create);
HOOK_DEF(hook_task_label_register);
HOOK_DEF(hook_task_label_start);
HOOK_DEF(hook_task_label_stop);
HOOK_DEF(hook_task_execute);
HOOK_DEF(hook_task_start);
HOOK_DEF(hook_task_end);
HOOK_DEF(hook_task_stop);
HOOK_DEF(hook_enter_busy_wait);
HOOK_DEF(hook_exit_busy_wait);
HOOK_DEF(hook_ss_pop);
HOOK_DEF(hook_ss_push);
HOOK_DEF(hook_ss_last);
HOOK_DEF(hook_ss_print);
HOOK_DEF(hook_ss_lock_client);
HOOK_DEF(hook_ss_lock_server);
HOOK_DEF(hook_flush);
HOOK_DEF(hook_mode_dead);
HOOK_DEF(hook_mode_runtime);
HOOK_DEF(hook_hwc);

/* The order is important */
struct hook_entry hook_list[] = {

	/* Threads */
	{ CLASS_ID_THREAD_CREATE,		hook_thread_create },
	{ CLASS_ID_EXTERNAL_THREAD_CREATE,	hook_ext_thread_create },
	{ CLASS_ID_THREAD_RESUME,		hook_thread_resume },
	{ CLASS_ID_EXTERNAL_THREAD_RESUME,	hook_thread_resume },
//	{ CLASS_ID_THREAD_SUSPEND,		hook_thread_suspend },
//	{ CLASS_ID_EXTERNAL_THREAD_SUSPEND,	hook_thread_suspend },
//	{ CLASS_ID_THREAD_SHUTDOWN,		hook_thread_suspend },
//	{ CLASS_ID_EXTERNAL_THREAD_SHUTDOWN,	hook_thread_suspend },
	{ CLASS_ID_WORKER_ENTER_BUSY_WAIT,	hook_enter_busy_wait },
	{ CLASS_ID_WORKER_EXIT_BUSY_WAIT,	hook_exit_busy_wait },

	/* Tasks creation */
	{ CLASS_ID_TC_TASK_CREATE_ENTER,	hook_task_create },
	{ CLASS_ID_OC_TASK_CREATE_ENTER,	hook_task_create },
	{ CLASS_ID_TASKFOR_INIT_ENTER,		hook_task_create },

	/* Tasks start/stop */
	{ CLASS_ID_TASK_START,			hook_task_start },
	{ CLASS_ID_TC_TASK_CREATE_ENTER,	hook_task_stop },
	{ CLASS_ID_TC_TASK_SUBMIT_EXIT,		hook_task_execute },
	{ CLASS_ID_TC_TASKWAIT_ENTER,		hook_task_stop },
	{ CLASS_ID_TC_TASKWAIT_EXIT,		hook_task_execute },
	{ CLASS_ID_TC_WAITFOR_ENTER,		hook_task_stop },
	{ CLASS_ID_TC_WAITFOR_EXIT,		hook_task_execute },
	{ CLASS_ID_TC_MUTEX_LOCK_ENTER,		hook_task_stop },
	{ CLASS_ID_TC_MUTEX_LOCK_EXIT,		hook_task_execute },
	{ CLASS_ID_TC_MUTEX_UNLOCK_ENTER,	hook_task_stop },
	{ CLASS_ID_TC_MUTEX_UNLOCK_EXIT,	hook_task_execute },
	{ CLASS_ID_TC_BLOCKING_API_BLOCK_ENTER,	hook_task_stop },
	{ CLASS_ID_TC_BLOCKING_API_BLOCK_EXIT,	hook_task_execute },
	{ CLASS_ID_TC_BLOCKING_API_UNBLOCK_ENTER, hook_task_stop },
	{ CLASS_ID_TC_BLOCKING_API_UNBLOCK_EXIT, hook_task_execute },
	{ CLASS_ID_TC_SPAWN_FUNCTION_ENTER,	hook_task_stop },
	{ CLASS_ID_TC_SPAWN_FUNCTION_EXIT,	hook_task_execute },
	{ CLASS_ID_TASK_BLOCK,			hook_task_stop },
	{ CLASS_ID_TASK_UNBLOCK,		hook_task_execute },

	/* Task label */
	{ CLASS_ID_TASK_LABEL,			hook_task_label_register },
	{ CLASS_ID_TASK_START,			hook_task_label_start },
	{ CLASS_ID_TASK_END,			hook_task_label_stop },
	{ CLASS_ID_TASK_UNBLOCK,		hook_task_label_start },
	{ CLASS_ID_TASK_BLOCK,			hook_task_label_stop },

	/* Task end (must be after task label) */
	{ CLASS_ID_TASK_END,			hook_task_end },

	/* Subsystem */
	{ CLASS_ID_THREAD_CREATE,		hook_ss_push },
	{ CLASS_ID_THREAD_SUSPEND,		hook_ss_print },
	{ CLASS_ID_THREAD_RESUME,		hook_ss_last },
	{ CLASS_ID_THREAD_SHUTDOWN,		hook_ss_print },
	{ CLASS_ID_EXTERNAL_THREAD_CREATE,	hook_ss_push },
	{ CLASS_ID_EXTERNAL_THREAD_SUSPEND,	hook_ss_print },
	{ CLASS_ID_EXTERNAL_THREAD_RESUME,	hook_ss_last },
	{ CLASS_ID_EXTERNAL_THREAD_SHUTDOWN,	hook_ss_print },

	{ CLASS_ID_TASK_END,			hook_ss_pop },
	{ CLASS_ID_TASK_START,			hook_ss_push },
	{ CLASS_ID_TC_TASKWAIT_ENTER,		hook_ss_push },
	{ CLASS_ID_TC_TASKWAIT_EXIT,		hook_ss_pop },

	{ CLASS_ID_TC_TASK_CREATE_ENTER,	hook_ss_push },
	{ CLASS_ID_TC_TASK_CREATE_EXIT,		hook_ss_pop },
//	{ CLASS_ID_TC_TASK_CREATE_EXIT,		hook_ss_push_arginit },
	{ CLASS_ID_OC_TASK_CREATE_ENTER,	hook_ss_push },
	{ CLASS_ID_OC_TASK_CREATE_EXIT,		hook_ss_pop },

	{ CLASS_ID_TC_TASK_SUBMIT_ENTER,	hook_ss_push },
	{ CLASS_ID_TC_TASK_SUBMIT_EXIT,		hook_ss_pop },
	{ CLASS_ID_OC_TASK_SUBMIT_ENTER,	hook_ss_push },
	{ CLASS_ID_OC_TASK_SUBMIT_EXIT,		hook_ss_pop },

	{ CLASS_ID_TC_WAITFOR_ENTER,		hook_ss_push },
	{ CLASS_ID_TC_WAITFOR_EXIT,		hook_ss_pop },
	{ CLASS_ID_TC_MUTEX_LOCK_ENTER,		hook_ss_push },
	{ CLASS_ID_TC_MUTEX_LOCK_EXIT,		hook_ss_pop },
	{ CLASS_ID_TC_MUTEX_UNLOCK_ENTER,	hook_ss_push },
	{ CLASS_ID_TC_MUTEX_UNLOCK_EXIT,	hook_ss_pop },
	{ CLASS_ID_TC_BLOCKING_API_BLOCK_ENTER,	hook_ss_push },
	{ CLASS_ID_TC_BLOCKING_API_BLOCK_EXIT,	hook_ss_pop },
	{ CLASS_ID_TC_BLOCKING_API_UNBLOCK_ENTER, hook_ss_push },
	{ CLASS_ID_TC_BLOCKING_API_UNBLOCK_EXIT, hook_ss_pop },
	{ CLASS_ID_OC_BLOCKING_API_UNBLOCK_ENTER, hook_ss_push },
	{ CLASS_ID_OC_BLOCKING_API_UNBLOCK_EXIT, hook_ss_pop },
	{ CLASS_ID_TC_SPAWN_FUNCTION_ENTER,	hook_ss_push },
	{ CLASS_ID_TC_SPAWN_FUNCTION_EXIT,	hook_ss_pop },
	{ CLASS_ID_OC_SPAWN_FUNCTION_ENTER,	hook_ss_push },
	{ CLASS_ID_OC_SPAWN_FUNCTION_EXIT,	hook_ss_pop },
	{ CLASS_ID_WORKER_ENTER_BUSY_WAIT,	hook_ss_push },
	{ CLASS_ID_WORKER_EXIT_BUSY_WAIT,	hook_ss_pop },
	{ CLASS_ID_DEPENDENCY_REGISTER_ENTER,	hook_ss_push },
	{ CLASS_ID_DEPENDENCY_REGISTER_EXIT,	hook_ss_pop },
	{ CLASS_ID_DEPENDENCY_UNREGISTER_ENTER,	hook_ss_push },
	{ CLASS_ID_DEPENDENCY_UNREGISTER_EXIT,	hook_ss_pop },
	{ CLASS_ID_SCHEDULER_ADD_TASK_ENTER,	hook_ss_push },
	{ CLASS_ID_SCHEDULER_ADD_TASK_EXIT,	hook_ss_pop },
	{ CLASS_ID_SCHEDULER_GET_TASK_ENTER,	hook_ss_push },
	{ CLASS_ID_SCHEDULER_GET_TASK_EXIT,	hook_ss_pop },

	{ CLASS_ID_SCHEDULER_LOCK_CLIENT,	hook_ss_lock_client },
	{ CLASS_ID_SCHEDULER_LOCK_SERVER,	hook_ss_lock_server },
//	{ CLASS_ID_SCHEDULER_LOCK_ASSIGN,	hook_ss_lock_assign },
	{ CLASS_ID_SCHEDULER_LOCK_SERVER_EXIT,	hook_ss_pop },

//	/* Thread id */
//	{ CLASS_ID_EXTERNAL_THREAD_CREATE,	hook_thread_create },
//	{ CLASS_ID_THREAD_CREATE,		hook_thread_create },
//	{ CLASS_ID_THREAD_RESUME,		hook_thread_resume },
//	{ CLASS_ID_EXTERNAL_THREAD_RESUME,	hook_thread_resume },
//	{ CLASS_ID_THREAD_SUSPEND,		hook_thread_suspend },
//	{ CLASS_ID_EXTERNAL_THREAD_SUSPEND,	hook_thread_suspend },
//	{ CLASS_ID_THREAD_SHUTDOWN,		hook_thread_suspend },
//	{ CLASS_ID_EXTERNAL_THREAD_SHUTDOWN,	hook_thread_suspend },

	/* Flush trace to disk */
	{ CLASS_ID_CTF_FLUSH,			hook_flush },

	/* Hardware counter hooks */
	{ CLASS_ID_THREAD_SUSPEND,			hook_hwc },
	{ CLASS_ID_THREAD_SHUTDOWN,			hook_hwc },
	{ CLASS_ID_TASK_START,				hook_hwc },
	{ CLASS_ID_TASK_END,				hook_hwc },
	{ CLASS_ID_TC_TASK_CREATE_ENTER,		hook_hwc },
	{ CLASS_ID_TC_TASK_SUBMIT_EXIT,			hook_hwc },
	{ CLASS_ID_TC_TASKWAIT_ENTER,			hook_hwc },
	{ CLASS_ID_TC_TASKWAIT_EXIT,			hook_hwc },
	{ CLASS_ID_TC_WAITFOR_ENTER,			hook_hwc },
	{ CLASS_ID_TC_WAITFOR_EXIT,			hook_hwc },
	{ CLASS_ID_TC_MUTEX_LOCK_ENTER,			hook_hwc },
	{ CLASS_ID_TC_MUTEX_LOCK_EXIT,			hook_hwc },
	{ CLASS_ID_TC_MUTEX_UNLOCK_ENTER,		hook_hwc },
	{ CLASS_ID_TC_MUTEX_UNLOCK_EXIT,		hook_hwc },
	{ CLASS_ID_TC_BLOCKING_API_BLOCK_ENTER,		hook_hwc },
	{ CLASS_ID_TC_BLOCKING_API_BLOCK_EXIT,		hook_hwc },
	{ CLASS_ID_TC_BLOCKING_API_UNBLOCK_ENTER,	hook_hwc },
	{ CLASS_ID_TC_BLOCKING_API_UNBLOCK_EXIT,	hook_hwc },
	{ CLASS_ID_TC_SPAWN_FUNCTION_ENTER,		hook_hwc },
	{ CLASS_ID_TC_SPAWN_FUNCTION_EXIT,		hook_hwc },

//	/* post */
	{ CLASS_ID_THREAD_SUSPEND,		hook_thread_suspend },
	{ CLASS_ID_THREAD_SHUTDOWN,		hook_thread_suspend },
	{ CLASS_ID_EXTERNAL_THREAD_SUSPEND,	hook_thread_suspend },
	{ CLASS_ID_EXTERNAL_THREAD_SHUTDOWN,	hook_thread_suspend },
//	{ CLASS_ID_THREAD_SUSPEND,		hook_thread_suspend },
//	{ CLASS_ID_TASK_BLOCK,			hook_task_block },
//	{ CLASS_ID_TASK_END,			hook_task_end },


	/* FIXME: We use this hack to exclude the serving tasks state
	 * from RM_RUNTIME mode, as in reality we are not doing useful
	 * work. */
	{ CLASS_ID_SCHEDULER_LOCK_SERVER,	hook_mode_dead },
	{ CLASS_ID_SCHEDULER_LOCK_SERVER_EXIT,	hook_mode_runtime },

	/* End */
	{ -1, NULL }
};

static void
hook_add(struct conv *conv, int class_id, hook_func_t func)
{
	int i;

	/* Keep one empty place at the end to set NULL */
	for(i=0; i<MAX_HOOKS - 1; i++)
	{
		if(conv->hook_table[class_id][i] != NULL)
			continue;

		conv->hook_table[class_id][i] = func;
		conv->hook_table[class_id][i+1] = NULL;
		return;
	}

	/* Too many hooks */
	err("too many hooks for class id %d\n", class_id);
	exit(EXIT_FAILURE);
}

double get_time()
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

/* Read the list of CPUs from the metadata and fill the physical CPU
 * (pcpu) array. Additionally, count how many CPUs are, and the maximum pcpu */
static void
populate_cpus(struct conv *conv, const char *cpu_list)
{
	char *saveptr, *p, *tmp;
	int pcpu, i;

	tmp = strdup(cpu_list);
	assert(tmp);

	p = strtok_r(tmp, ",", &saveptr);

	conv->ncpus = 0;

	while(p)
	{
		pcpu = atoi(p);
		assert(pcpu >= 0);

		if(pcpu >= MAX_CPUS)
		{
			err("too many cpus\n");
			exit(EXIT_FAILURE);
		}

		i = conv->pcpu_index[pcpu];

		if(i != -1)
		{
			err("repeated cpus\n");
			exit(EXIT_FAILURE);
		}

		conv->cpus[conv->ncpus].pcpu = pcpu;
		conv->pcpu_index[pcpu] = conv->ncpus;
		conv->ncpus++;
		if(pcpu > conv->max_pcpu)
			conv->max_pcpu = pcpu;

		p = strtok_r(NULL, ",", &saveptr);
	}

	assert(conv->max_pcpu >= 0);
	assert(conv->ncpus > 0);

	free(tmp);
}

/*
 * Initializes the sink component.
 */
bt_component_class_initialize_method_status conv_initialize(
	bt_self_component_sink *self_component_sink,
	bt_self_component_sink_configuration *configuration,
	const bt_value *params, void *initialize_method_data)
{
	assert(params);
	assert(bt_value_is_map(params));
	assert(bt_value_map_has_entry(params, "output_dir"));

	/* Allocate a private data structure */
	struct conv *conv = malloc(sizeof(*conv));

	/* Initialize the first event message's index */
	conv->index = 1;

	/* Set counter to 0 */
	conv->dummy = 0;
	conv->ev_processed = 0;
	conv->ev_ignored = 0;
	conv->ev_emitted = 0;
	conv->ev_last = 0;
	conv->ev_progress = 0;

	conv->last_reported = 0;
	conv->last_written = 0;

	conv->threads = NULL;
	conv->tasks = NULL;
	conv->task_types = NULL;

	int i;
	for(i=0; i<MAX_ID; i++)
	{
		conv->hook_table[i][0] = NULL;
	}

	for(i=0; i<MAX_ID; i++)
	{
		conv->ss_table[i] = -1;
	}


	for(i=0; i<MAX_CPUS; i++)
	{
		conv->cpus[i].thread = NULL;
		conv->cpus[i].pcpu = -1;
		conv->pcpu_index[i] = -1;
	}

	conv->tic = get_time();
	conv->t0 = conv->tic;
	conv->curr_cpu = -1;
	conv->max_pcpu = -1;

	conv->n_acc_ev = 0;
	conv->last_thread_color = 1;

	/* Unknown at init time */
	conv->ncpus = -1;
	conv->nvcpus = -1;

	/* HWC init to not checked */
	conv->hwc_enabled = -1;
	conv->nhwc = 0;

	/* Set to 0 the clock offset, start and end times */
	conv->clock_offset_ns = 0;
	conv->clock_start_ns = 0;
	conv->clock_end_ns = 0;

	conv->external_threads = 0;

	/* Init hook table */
	for(i=0; hook_list[i].func != NULL; i++)
		hook_add(conv, hook_list[i].class_id, hook_list[i].func);

	/* Init subsystem table */
	for(i=0; ss_list[i][0] != -1; i++)
		conv->ss_table[ss_list[i][0]] = ss_list[i][1];

	const bt_value* split_value = bt_value_map_borrow_entry_value_const(
			params, "split_events");

	assert(split_value);
	assert(bt_value_is_bool(split_value));

	conv->split_events = bt_value_bool_get(split_value);

	const bt_value* quiet_value = bt_value_map_borrow_entry_value_const(
			params, "quiet");

	assert(quiet_value);
	assert(bt_value_is_bool(quiet_value));

	conv->quiet = bt_value_bool_get(quiet_value);

	const bt_value *filters = bt_value_map_borrow_entry_value_const(params, "filters");
	assert(filters);
	assert(bt_value_is_array(filters));

	conv->nfilters = bt_value_array_get_length(filters);

	for (int i = 0; i < conv->nfilters; ++i) {
		const bt_value *event = bt_value_array_borrow_element_by_index_const(filters, i);
		assert(event);
		assert(bt_value_is_signed_integer(event));
		conv->filters[i] = bt_value_integer_signed_get(event);
	}

	const bt_value* path_value = bt_value_map_borrow_entry_value_const(
			params, "output_dir");

	assert(path_value);
	assert(bt_value_is_string(path_value));

	conv->output_dir = bt_value_string_get(path_value);
	assert(conv->output_dir);

	char prv_path[PATH_MAX];

	if(snprintf(prv_path, PATH_MAX, "%s/%s.prv",
				conv->output_dir, TRACE_NAME) >= PATH_MAX)
	{
		err("prv file path too large\n");
		exit(EXIT_FAILURE);
	}

	if((conv->prv = fopen(prv_path, "w")) == NULL)
	{
		perror("opening output prv file");
		exit(EXIT_FAILURE);
	}

	char pcf_path[PATH_MAX];

	if(snprintf(pcf_path, PATH_MAX, "%s/%s.pcf",
				conv->output_dir, TRACE_NAME) >= PATH_MAX)
	{
		err("pcf file path too large\n");
		exit(EXIT_FAILURE);
	}

	if(snprintf(conv->row_file_path, PATH_MAX, "%s/%s.row",
				conv->output_dir, TRACE_NAME) >= PATH_MAX)
	{
		err("row file path too large\n");
		exit(EXIT_FAILURE);
	}

	pcf_init(&conv->pcf);

	if((conv->pcf_file = fopen(pcf_path, "w")) == NULL)
	{
		perror("opening output pcf file");
		exit(EXIT_FAILURE);
	}

	/* Print the prv header */
	fprintf(conv->prv, PRV_HEADER_FMT, 0, 0);

	/* Set the component's user data to our private data structure */
	bt_self_component_set_data(
		bt_self_component_sink_as_self_component(self_component_sink),
		conv);

	/*
	 * Add an input port named `in` to the sink component.
	 *
	 * This is needed so that this sink component can be connected to a
	 * filter or a source component. With a connected upstream
	 * component, this sink component can create a message iterator
	 * to consume messages.
	 */
	bt_self_component_sink_add_input_port(self_component_sink,
		"in", NULL, NULL);

	return BT_COMPONENT_CLASS_INITIALIZE_METHOD_STATUS_OK;
}

/* We don't know the number of "threads" until we had read all the events, so we
 * fix the header at the end. */
static void
fix_header(struct conv *conv)
{
	/* Go to the first byte */
	fseek(conv->prv, 0, SEEK_SET);

	/* And print the whole header again */
	fprintf(conv->prv, PRV_HEADER_FMT, 0, conv->ncpus + conv->nvcpus);
}

void
write_row_file(struct conv *conv)
{
	FILE *f;
	int i;

	f = fopen(conv->row_file_path, "w");

	if(f == NULL)
	{
		perror("cannot open row file");
		exit(EXIT_FAILURE);
	}

	fprintf(f, "LEVEL NODE SIZE 1\n");
	fprintf(f, "hostname\n");
	fprintf(f, "\n");

	assert(conv->nvcpus >= 1);
	assert(conv->nvcpus == conv->external_threads);
	fprintf(f, "LEVEL THREAD SIZE %d\n", conv->ncpus + conv->nvcpus);

	for(i=0; i<conv->ncpus; i++)
		fprintf(f, "CPU %2d\n", conv->cpus[i].pcpu);

	/* The first virtual cpu is always the leader thread */
	fprintf(f, "LEADER\n");

	for(i=0; i<conv->nvcpus-1; i++)
		fprintf(f, "EXT %d\n", i);

	fclose(f);
}


/*
 * Finalizes the sink component.
 */
void conv_finalize(bt_self_component_sink *self_component_sink)
{
	/* Retrieve our private data from the component's user data */
	struct conv *conv = bt_self_component_get_data(
	bt_self_component_sink_as_self_component(self_component_sink));

	double dt;

	fix_header(conv);

	if(fclose(conv->prv) != 0)
	{
		perror("error closing prv file");
		exit(EXIT_FAILURE);
	}

	/* Set the task types before writing the PCF file */
	pcf_set_task_types(&conv->pcf, conv->task_types);

	pcf_write(&conv->pcf, conv->pcf_file);

	if(fclose(conv->pcf_file) != 0)
	{
		perror("error closing pcf file");
		exit(EXIT_FAILURE);
	}

	write_row_file(conv);

	dt = get_time() - conv->t0;
	if(!conv->quiet)
	{
		err("\ntotal events: %lu in, %lu out, avg speed %.1f kev/s\n",
				conv->ev_processed,
				conv->ev_emitted,
				(double) conv->ev_processed / dt / 1e3);
	}

	//printf("dummy = %"PRIu64"\n", conv->dummy);

	//int i;
	//for(i=0; i<MAX_ID; i++)
	//{
	//	if(evname[i] != NULL)
	//	{
	//		printf("%d  %s  %lu\n", i, evname[i], evcount[i]);
	//	}
	//}

	/* Free the allocated structure */
	free(conv);
}

/*
 * Called when the trace processing graph containing the sink component
 * is configured.
 *
 * This is where we can create our upstream message iterator.
 */
bt_component_class_sink_graph_is_configured_method_status
conv_graph_is_configured(bt_self_component_sink *self_component_sink)
{
	/* Retrieve our private data from the component's user data */
	struct conv *conv = bt_self_component_get_data(
			bt_self_component_sink_as_self_component(
				self_component_sink));

	/* Borrow our unique port */
	bt_self_component_port_input *in_port =
	bt_self_component_sink_borrow_input_port_by_index(
			self_component_sink, 0);

	/* Create the uptream message iterator */
	bt_message_iterator_create_from_sink_component(
			self_component_sink, in_port,
			&conv->message_iterator);

	return BT_COMPONENT_CLASS_SINK_GRAPH_IS_CONFIGURED_METHOD_STATUS_OK;
}

uint64_t
get_event_external_tid(struct conv *conv, const struct bt_event *event)
{
	/* External threads have their tid in the unbounded struct. We
	 * can determine if the current event refers to a unbounded CPU
	 * by looking at the curr CPU index */

	const bt_field *unbounded, *ctx, *tid_field;

	ctx = bt_event_borrow_common_context_field_const(event);
	assert(ctx);

	unbounded = bt_field_structure_borrow_member_field_by_name_const(
			ctx, "unbounded");

	assert(unbounded);

	tid_field = bt_field_structure_borrow_member_field_by_name_const(
			unbounded, "tid");

	assert(tid_field);


	uint64_t tid = bt_field_integer_unsigned_get_value(
			tid_field);

	return tid;
}

uint64_t
get_event_tid(struct conv *conv, const struct bt_event *event)
{
	/* External threads have their tid in the unbounded struct. We
	 * can determine if the current event refers to a unbounded CPU
	 * by looking at the curr CPU index */

	const bt_field *payload, *unbounded, *ctx, *tid_field;

	if(conv->curr_cpu >= conv->ncpus)
	{
		/* Virtual CPU */
		ctx = bt_event_borrow_common_context_field_const(event);
		assert(ctx);
		unbounded =
			bt_field_structure_borrow_member_field_by_name_const(
					ctx, "unbounded");
		assert(unbounded);

		tid_field =
			bt_field_structure_borrow_member_field_by_name_const(
					unbounded, "tid");
		assert(tid_field);
	}
	else
	{
		/* Physical CPU */
		payload = bt_event_borrow_payload_field_const(event);

		assert(payload);

		tid_field =
			bt_field_structure_borrow_member_field_by_name_const(
					payload, "tid");
		assert(tid_field);
	}


	uint64_t tid = bt_field_integer_unsigned_get_value(
			tid_field);

	return tid;
}

static const bt_field *
get_field(const struct bt_event *event, const char *name)
{
	const bt_field *payload_field =
		bt_event_borrow_payload_field_const(event);

	assert(payload_field);

	const bt_field *field =
		bt_field_structure_borrow_member_field_by_name_const(
			payload_field, name);

	return field;
}

static const char *
get_field_str(const struct bt_event *event, const char *name)
{
	const bt_field *field = get_field(event, name);

	assert(field);

	return bt_field_string_get_value(field);
}

static uint64_t
get_field_uint64(const struct bt_event *event, const char *name)
{
	const bt_field *field = get_field(event, name);

	assert(field);

	return (uint64_t) bt_field_integer_unsigned_get_value(field);
}

static int64_t
get_field_int64(const struct bt_event *event, const char *name)
{
	const bt_field *field = get_field(event, name);

	assert(field);

	return (uint64_t) bt_field_integer_signed_get_value(field);
}

static inline int
filter_allowed(struct conv *conv, long long type)
{
	if (!conv->nfilters)
		return 1;

	for (int i = 0; i < conv->nfilters; ++i) {
		if (type == (long long) conv->filters[i])
			return 1;
	}

	return 0;
}

void
add_prv_ev(struct conv *conv, long long type, long long value)
{
	int i;
	if(conv->n_acc_ev + 1 >= MAX_ACC_EVENTS)
	{
		err("too many acc events\n");
		exit(EXIT_FAILURE);
	}

	i = conv->n_acc_ev;

	if (!filter_allowed(conv, type))
		return;

	conv->acc_ev[i].type = type;
	conv->acc_ev[i].value = value;

	conv->n_acc_ev++;
}

struct thread *
get_thread(struct conv *conv, uint64_t tid)
{
	struct thread *t = NULL;

	HASH_FIND(hh, conv->threads, &tid, sizeof(t->tid), t);
	return t;
}

void
hook_ext_thread_create(struct conv *conv, const struct bt_event *event, int class_id)
{
	dbg("ext_thread_create: class_id=%d\n", class_id);
	uint64_t tid = get_event_tid(conv, event);
	struct thread *thread;

	/* It could be created before */
	HASH_FIND(hh, conv->threads, &tid, sizeof(tid), thread);

	if(thread)
	{
		assert(thread->external);
		assert(thread->tid == tid);
		assert(thread->state == THREAD_ST_UNKNOWN);
		thread->state = THREAD_ST_CREATED;
	}
	else
	{
		thread = malloc(sizeof(*thread));
		if(!thread) exit(1);

		thread->tid = tid;
		thread->cpu = conv->curr_cpu;
		thread->external = 1;
		thread->state = THREAD_ST_CREATED;
		thread->task = NULL;
		thread->busy_wait = 0;
		thread->color = conv->last_thread_color++;
		thread->n_stack = 0;

		HASH_ADD(hh, conv->threads, tid, sizeof(thread->tid), thread);
	}

	conv->cpus[conv->curr_cpu].thread = thread;

	add_prv_ev(conv, EV_TYPE_RUNTIME_CODE, RA_RUNTIME);
	add_prv_ev(conv, EV_TYPE_RUNNING_THREAD_TID, thread->color);
}

void
hook_thread_create(struct conv *conv, const struct bt_event *event, int class_id)
{
	dbg("thread_create: class_id=%d\n", class_id);
	uint64_t tid = get_event_tid(conv, event);
	struct thread *thread;

	thread = conv->cpus[conv->curr_cpu].thread;

	if(thread)
	{
		/* We can only had it created if is external */
		assert(thread->external);
		assert(thread->tid == tid);
		assert(thread->state == THREAD_ST_UNKNOWN);

		thread->state = THREAD_ST_CREATED;
	}
	else
	{
		HASH_FIND(hh, conv->threads, &tid, sizeof(tid), thread);
		assert(!thread);

		thread = malloc(sizeof(*thread));
		if(!thread) exit(1);

		thread->tid = tid;
		thread->cpu = conv->curr_cpu;
		thread->external = 0;
		thread->state = THREAD_ST_CREATED;
		thread->task = NULL;
		thread->busy_wait = 0;
		thread->color = conv->last_thread_color++;
		thread->n_stack = 0;

		HASH_ADD(hh, conv->threads, tid, sizeof(thread->tid), thread);
	}

	conv->cpus[conv->curr_cpu].thread = thread;

	add_prv_ev(conv, EV_TYPE_RUNTIME_CODE, RA_RUNTIME);
	add_prv_ev(conv, EV_TYPE_RUNNING_THREAD_TID, thread->color);
}

void
hook_thread_resume(struct conv *conv, const struct bt_event *event, int class_id)
{
	int i;
	dbg("thread_resume: class_id=%d\n", class_id);
	dbg("thread_resume: curr_cpu=%d\n", conv->curr_cpu);
	uint64_t tid = get_event_tid(conv, event);
	struct thread *thread = get_thread(conv, tid);
	assert(thread);
	dbg("thread_resume: thread=%p\n", thread);

	assert(thread->state != THREAD_ST_RESUMED);

	if(thread->state != THREAD_ST_CREATED && thread->external == 0)
	{
		assert(conv->cpus[conv->curr_cpu].thread == NULL);
	}

	/* Set the thread running in the current CPU */
	conv->cpus[conv->curr_cpu].thread = thread;

	/* Set the current CPU as the one running the thread */
	thread->state = THREAD_ST_RESUMED;
	thread->cpu = conv->curr_cpu;

	add_prv_ev(conv, EV_TYPE_RUNTIME_CODE, RA_RUNTIME);
	add_prv_ev(conv, EV_TYPE_RUNNING_THREAD_TID, thread->color);
	add_prv_ev(conv, EV_TYPE_RUNTIME_MODE, RM_RUNTIME);

	if(thread->busy_wait)
		add_prv_ev(conv, EV_TYPE_RUNTIME_BUSYWAITING, RA_BUSYWAITING);

	if(thread->task && thread->task->state == TASK_ST_RUNNING)
		hook_task_execute(conv, event, class_id);

	/* Reset the counters if we have HWC */
	/* FIXME: We don't emit events before the first HWC struct is
	 * found, so we could miss some thread_resume events */
	if(conv->hwc_enabled == 1)
	{
		for(i=0; i<conv->nhwc; i++)
			add_prv_ev(conv, conv->hwc_table[i], 0);
	}
}

void
hook_thread_suspend(struct conv *conv, const struct bt_event *event, int class_id)
{
	dbg("thread_suspend: class_id=%d\n", class_id);
	dbg("thread_suspend: curr_cpu=%d\n", conv->curr_cpu);
	uint64_t tid = get_event_tid(conv, event);
	struct thread *thread = get_thread(conv, tid);
	assert(thread);
	dbg("thread_suspend: thread=%p\n", thread);

	/* Remove the CPU running the thread */
	thread->state = THREAD_ST_SUSPENDED;
	if(!thread->external)
	{
		thread->cpu = -1;

		/* Remove the thread from the current CPU */
		assert(conv->cpus[conv->curr_cpu].thread == thread);
		conv->cpus[conv->curr_cpu].thread = NULL;
	}

	add_prv_ev(conv, EV_TYPE_RUNTIME_CODE, RA_END);
	add_prv_ev(conv, EV_TYPE_RUNNING_THREAD_TID, RA_END);

	if(thread->busy_wait)
		add_prv_ev(conv, EV_TYPE_RUNTIME_BUSYWAITING, RA_END);

	if(thread->task && thread->task->state == TASK_ST_RUNNING)
		hook_task_stop(conv, event, class_id);

	/* FIXME: The task stop may emit a RM_RUNTIME event, which will
	 * be duplicated with this one: */
	add_prv_ev(conv, EV_TYPE_RUNTIME_MODE, RM_DEAD);
}

void
hook_enter_busy_wait(struct conv *conv, const struct bt_event *event, int class_id)
{
	dbg("enter_busy_wait: class_id=%d\n", class_id);
	dbg("enter_busy_wait: curr_cpu=%d\n", conv->curr_cpu);
	struct thread *thread = conv->cpus[conv->curr_cpu].thread;
	assert(thread);
	dbg("enter_busy_wait: thread=%p\n", thread);

	thread->busy_wait = 1;

	add_prv_ev(conv, EV_TYPE_RUNTIME_BUSYWAITING, RA_BUSYWAITING);
}

void
hook_exit_busy_wait(struct conv *conv, const struct bt_event *event, int class_id)
{
	dbg("exit_busy_wait: class_id=%d\n", class_id);
	dbg("exit_busy_wait: curr_cpu=%d\n", conv->curr_cpu);
	struct thread *thread = conv->cpus[conv->curr_cpu].thread;
	assert(thread);
	dbg("exit_busy_wait: thread=%p\n", thread);

	thread->busy_wait = 0;

	add_prv_ev(conv, EV_TYPE_RUNTIME_BUSYWAITING, RA_END);
}

void
hook_task_label_register(struct conv *conv, const struct bt_event *event, int class_id)
{
	struct task_type *tt = NULL;

	const char *label, *srcline;
	uint64_t type;

	dbg("task_label_register: class_id=%d\n", class_id);

	label = get_field_str(event, "label");
	srcline = get_field_str(event, "source");
	type = get_field_uint64(event, "type");

	dbg("task_label_register: type=%lu label=[%s] srcline=[%s]\n",
			type, label, srcline);

	HASH_FIND(hh, conv->task_types, &type, sizeof(type), tt);

	if(tt)
	{
		err("The task type %lu was already in the hash table\n", type);
		exit(EXIT_FAILURE);
	}

	/* Register the task type */
	tt = malloc(sizeof(*tt));
	if(!tt) abort();

	tt->type = type;
	strncpy(tt->label, label, sizeof(tt->label) - 1);
	strncpy(tt->srcline, srcline, sizeof(tt->srcline) - 1);
	tt->label[sizeof(tt->label) - 1] = '\0';
	tt->srcline[sizeof(tt->srcline) - 1] = '\0';

	HASH_ADD(hh, conv->task_types, type, sizeof(type), tt);
}

void
hook_task_label_start(struct conv *conv, const struct bt_event *event, int class_id)
{
	dbg("task_label_start: class_id=%d\n", class_id);
	struct thread *thread = conv->cpus[conv->curr_cpu].thread;
	assert(thread);
	struct task *task = thread->task;
	assert(task);

	add_prv_ev(conv, EV_TYPE_RUNNING_TASK_LABEL, task->type);
	add_prv_ev(conv, EV_TYPE_RUNNING_TASK_SOURCE, task->type);
}

void
hook_task_label_stop(struct conv *conv, const struct bt_event *event, int class_id)
{
	dbg("task_label_stop: class_id=%d\n", class_id);
	struct thread *thread = conv->cpus[conv->curr_cpu].thread;
	assert(thread);
	struct task *task = thread->task;
	assert(task);

	add_prv_ev(conv, EV_TYPE_RUNNING_TASK_LABEL, RA_END);
	add_prv_ev(conv, EV_TYPE_RUNNING_TASK_SOURCE, RA_END);
}

void
hook_task_create(struct conv *conv, const struct bt_event *event, int class_id)
{
	struct task *task = NULL;

	uint64_t type, id;

	dbg("task_create: class_id=%d\n", class_id);

	type = get_field_uint64(event, "type");
	id = get_field_uint64(event, "id");

	dbg("task_create: id=%lu type=%lu\n", id, type);

	HASH_FIND(hh, conv->tasks, &id, sizeof(id), task);

	if(task)
	{
		err("The task with id %lu was already in the hash table\n", id);
		exit(EXIT_FAILURE);
	}

	/* Register the task */
	task = malloc(sizeof(*task));
	if(!task) abort();

	task->id = id;
	task->type = type;
	task->state = TASK_ST_UNINIT;

	HASH_ADD(hh, conv->tasks, id, sizeof(id), task);
}

void
hook_task_start(struct conv *conv, const struct bt_event *event, int class_id)
{
	uint64_t id;
	struct task *task = NULL;

	dbg("task_start: class_id=%d\n", class_id);

	struct thread *thread = conv->cpus[conv->curr_cpu].thread;
	assert(thread);

	id = get_field_uint64(event, "id");

	HASH_FIND(hh, conv->tasks, &id, sizeof(id), task);

	if(!task)
	{
		err("The task with id %lu is missing in the hash table\n", id);
		exit(EXIT_FAILURE);
	}

	dbg("task_start: id=%lu\n", task->id);

	assert(thread->task == NULL);
	thread->task = task;

	dbg("task_start: thread=%p task=%p\n", thread, task);
	dbg("task_start: cpu=%d\n", conv->curr_cpu);

	hook_task_execute(conv, event, class_id);
}

void
hook_task_execute(struct conv *conv, const struct bt_event *event, int class_id)
{
	dbg("task_execute: class_id=%d\n", class_id);
	struct thread *thread = conv->cpus[conv->curr_cpu].thread;
	assert(thread);
	dbg("task_execute: thread=%p\n", thread);
	dbg("task_execute: cpu=%d\n", conv->curr_cpu);
	struct task *task = thread->task;
	assert(task);

	add_prv_ev(conv, EV_TYPE_RUNTIME_TASKS, RA_TASK);
	add_prv_ev(conv, EV_TYPE_RUNNING_TASK_ID, task->id);

	/* Avoid a task mode event when we are in waitfor */
	if(!(class_id == CLASS_ID_THREAD_RESUME &&
			thread->ev_stack[thread->n_stack-1] == RS_WAIT_FOR))
	{
		add_prv_ev(conv, EV_TYPE_RUNTIME_MODE, RM_TASK);
	}
}

void
hook_task_stop(struct conv *conv, const struct bt_event *event, int class_id)
{
	dbg("task_stop: class_id=%d\n", class_id);
	add_prv_ev(conv, EV_TYPE_RUNTIME_TASKS, RA_END);
	add_prv_ev(conv, EV_TYPE_RUNNING_TASK_ID, RA_END);

	/* FIXME: too complex */
	if(class_id == CLASS_ID_THREAD_SUSPEND ||
			class_id == CLASS_ID_EXTERNAL_THREAD_SUSPEND)
	{
		add_prv_ev(conv, EV_TYPE_RUNTIME_MODE, RM_DEAD);
	}
	else
	{
		add_prv_ev(conv, EV_TYPE_RUNTIME_MODE, RM_RUNTIME);
	}
}

void
hook_task_end(struct conv *conv, const struct bt_event *event, int class_id)
{
	dbg("task_end: class_id=%d\n", class_id);
	struct thread *thread = conv->cpus[conv->curr_cpu].thread;
	assert(thread);

	thread->task = NULL;

	hook_task_stop(conv, event, class_id);
}

void
hook_flush(struct conv *conv, const struct bt_event *event, int class_id)
{
	int prv_cpu;
	int64_t t0, t1;

	t0 = get_field_uint64(event, "start");
	t1 = get_field_uint64(event, "end");

	/* Adjust the time value using the clock offset */
	t0 += conv->clock_offset_ns;
	t1 += conv->clock_offset_ns;

	assert(t0 >= 0);
	assert(t1 >= 0);

	prv_cpu = conv->curr_cpu + 1;

	/* Why don't we receive flush enter and flush exit (?) */
	fprintf(conv->prv,
			"2:0:1:1:%d:%ld:%d:%d\n",
			prv_cpu, t0, EV_TYPE_CTF_FLUSH, 1);
	fprintf(conv->prv,
			"2:0:1:1:%d:%ld:%d:%d\n",
			prv_cpu, t1, EV_TYPE_CTF_FLUSH, 0);
}

static void
detect_hwc(struct conv *conv, const struct bt_event *event, int class_id)
{
	const bt_field_class* hwc_class;
	const bt_field *context_field, *hwc_field;
	const bt_field_class_structure_member *member_class;
	uint64_t i, j, ncounters;
	const char *member_name;
	struct hwc *hwc_entry;

	context_field = bt_event_borrow_specific_context_field_const(event);

	if(context_field == NULL)
	{
		conv->hwc_enabled = 0;
		return;
	}


	hwc_field = bt_field_structure_borrow_member_field_by_name_const(
			context_field, "hwc");

	if(hwc_field == NULL)
	{
		conv->hwc_enabled = 0;
		return;
	}

	conv->hwc_enabled = 1;

	/* Read the HWC list */
	hwc_class = bt_field_borrow_class_const(hwc_field);
	ncounters = bt_field_class_structure_get_member_count(hwc_class);

	if(ncounters > MAX_HWC)
	{
		err("too many HW counters (%ld)\n", ncounters);
		exit(EXIT_FAILURE);
	}

	conv->nhwc = ncounters;
	dbg("loaded %ld counters\n", ncounters);

	/* Build the hwc table */
	for(i=0; i<conv->nhwc; i++)
	{
		member_class = bt_field_class_structure_borrow_member_by_index_const(
				hwc_class, i);

		assert(member_class);
		member_name = bt_field_class_structure_member_get_name(
				member_class);

		assert(member_name);

		hwc_entry = NULL;

		for(j=0; hwc_table[j].name != NULL; j++)
		{
			if(strcmp(hwc_table[j].name, member_name) == 0)
			{
				hwc_entry = &hwc_table[j];
				break;
			}
		}

		if(hwc_entry == NULL)
		{
			err("unknown HWC %s\n", member_name);
			exit(EXIT_FAILURE);
		}

		conv->hwc_table[i] = hwc_entry->id;
		dbg("added counter: %s with id %ld\n", member_name, hwc_entry->id);
	}
}

static void
emit_hwc(struct conv *conv, const struct bt_event *event, int class_id)
{
	const bt_field *context_field, *hwc_field, *counter_field;
	uint64_t i, hwc_delta;
	int hwc_id;

	context_field = bt_event_borrow_specific_context_field_const(event);

	assert(context_field);

	hwc_field = bt_field_structure_borrow_member_field_by_name_const(
			context_field, "hwc");

	assert(hwc_field);

	for(i=0; i<conv->nhwc; i++)
	{
		/* Read member by index */
		counter_field = bt_field_structure_borrow_member_field_by_index_const(
				hwc_field, i);

		hwc_delta = bt_field_integer_unsigned_get_value(counter_field);
		hwc_id = conv->hwc_table[i];

		add_prv_ev(conv, hwc_id, hwc_delta);
	}
}


void
hook_hwc(struct conv *conv, const struct bt_event *event, int class_id)
{
	if(conv->hwc_enabled == -1)
		detect_hwc(conv, event, class_id);

	/* No HW counters */
	if(conv->hwc_enabled == 0) return;

	emit_hwc(conv, event, class_id);
}

void
hook_ss_lock_client(struct conv *conv, const struct bt_event *event, int class_id)
{
	int prv_cpu;
	int64_t ts;

	ts = (int64_t) get_field_uint64(event, "ts_acquire");
	ts += conv->clock_offset_ns;

	assert(ts >= 0);

	prv_cpu = conv->curr_cpu + 1;

	/* FIXME: Ultra hack: send a event in the past, when the lock
	 * was acquired */
	fprintf(conv->prv,
			"2:0:1:1:%d:%ld:%d:%d\n",
			prv_cpu, ts,
			EV_TYPE_RUNTIME_SUBSYSTEMS,
			RS_SCHEDULER_LOCK_ENTER);

	/* Then print the last event in the stack */
	hook_ss_last(conv, event, class_id);
}

void
hook_ss_lock_server(struct conv *conv, const struct bt_event *event, int class_id)
{
	int prv_cpu;
	int64_t ts;

	ts = (int64_t) get_field_int64(event, "ts_acquire");
	ts += conv->clock_offset_ns;

	assert(ts >= 0);

	prv_cpu = conv->curr_cpu + 1;

	/* FIXME: Ultra hack: send a event in the past, when the lock
	 * was acquired */
	fprintf(conv->prv,
			"2:0:1:1:%d:%ld:%d:%d\n",
			prv_cpu, ts,
			EV_TYPE_RUNTIME_SUBSYSTEMS,
			RS_SCHEDULER_LOCK_ENTER);

	/* Add to stack this lock serving */
	hook_ss_push(conv, event, class_id);
}

void
hook_ss_print(struct conv *conv, const struct bt_event *event, int class_id)
{
	int subsystem = conv->ss_table[class_id];

	/* It must be defined in ss_list */
	assert(subsystem != -1);

	add_prv_ev(conv, EV_TYPE_RUNTIME_SUBSYSTEMS, subsystem);
}

void
hook_ss_last(struct conv *conv, const struct bt_event *event, int class_id)
{
	struct thread *thread = conv->cpus[conv->curr_cpu].thread;
	assert(thread);

	dbg("hook_ss_last: class_id=%d curr_cpu=%d thread=%p\n",
			class_id, conv->curr_cpu, thread);

	assert(thread->n_stack >= 1);

	add_prv_ev(conv, EV_TYPE_RUNTIME_SUBSYSTEMS,
			thread->ev_stack[thread->n_stack-1]);
}

void
hook_ss_pop(struct conv *conv, const struct bt_event *event, int class_id)
{
	struct thread *thread = conv->cpus[conv->curr_cpu].thread;
	assert(thread);

	dbg("hook_ss_pop: class_id=%d curr_cpu=%d thread=%p\n",
			class_id, conv->curr_cpu, thread);

	assert(thread->n_stack >= 2);

	thread->n_stack--;

	//if(thread->n_stack == 0)
	//	add_prv_ev(conv, EV_TYPE_RUNTIME_SUBSYSTEMS, 0);
	//else

	add_prv_ev(conv, EV_TYPE_RUNTIME_SUBSYSTEMS,
			thread->ev_stack[thread->n_stack-1]);
}

void
hook_ss_push(struct conv *conv, const struct bt_event *event, int class_id)
{
	struct thread *thread = conv->cpus[conv->curr_cpu].thread;
	assert(thread);

	dbg("hook_ss_push: class_id=%d curr_cpu=%d thread=%p\n",
			class_id, conv->curr_cpu, thread);

	if(thread->n_stack + 1 > MAX_EV_STACK)
	{
		err("too many events stacked\n");
		exit(EXIT_FAILURE);
	}

	int subsystem = conv->ss_table[class_id];

	/* It must be defined in ss_list */
	assert(subsystem != -1);

	thread->ev_stack[thread->n_stack++] = subsystem;
	add_prv_ev(conv, EV_TYPE_RUNTIME_SUBSYSTEMS, subsystem);
}

void
hook_mode_dead(struct conv *conv, const struct bt_event *event, int class_id)
{
	add_prv_ev(conv, EV_TYPE_RUNTIME_MODE, RM_DEAD);
}

void
hook_mode_runtime(struct conv *conv, const struct bt_event *event, int class_id)
{
	add_prv_ev(conv, EV_TYPE_RUNTIME_MODE, RM_RUNTIME);
}

static void
flush_acc_events(struct conv *conv, const struct bt_message *message)
{
	int i;
	int64_t ns;
	const bt_clock_snapshot *clock_snapshot =
		bt_message_event_borrow_default_clock_snapshot_const(
				message);

	/* Read the clock applying the offset */
	bt_clock_snapshot_get_ns_from_origin(clock_snapshot, &ns);

	/* Paraver begins CPUs at 1 */
	int prv_cpu = conv->curr_cpu + 1;

	/* If split_events is not enabled, multiple events happening at
	 * the same time are printed in the same line, saving space, but
	 * increasing the complexity to be processed as a stream with
	 * common text tools. Otherwise, if enabled, only one event per
	 * line is printed, which is good for grep(1). */
	if(conv->split_events)
	{
		for(i=0; i<conv->n_acc_ev; i++)
		{
			fprintf(conv->prv,
					"2:0:1:1:%d:%ld:%lld:%lld\n",
					prv_cpu, ns,
					conv->acc_ev[i].type,
					conv->acc_ev[i].value);

		}
	}
	else
	{
		fprintf(conv->prv, "2:0:1:1:%d:%ld", prv_cpu, ns);

		for(i=0; i<conv->n_acc_ev; i++)
		{
			fprintf(conv->prv,
					":%lld:%lld", conv->acc_ev[i].type,
					conv->acc_ev[i].value);
		}

		fprintf(conv->prv, "\n");
	}

	conv->ev_emitted += conv->n_acc_ev;
	conv->n_acc_ev = 0;
	conv->ev_progress = (double) (ns - conv->clock_start_ns) /
		(conv->clock_end_ns - conv->clock_start_ns);
}

/* External threads at not bounded to any CPU so we cannot determine
 * where they are running. So we assign them to one fake CPU each, the
 * virtual CPU, as soon as we see any running */
static int
get_external_thread_cpu(struct conv *conv, const bt_event *event)
{
	struct thread *thread = NULL;
	uint64_t tid = get_event_external_tid(conv, event);

	/* Try to find the thread */
	HASH_FIND(hh, conv->threads, &tid, sizeof(thread->tid), thread);

	/* Add the thread if not found */
	if(thread == NULL)
	{
		thread = malloc(sizeof(*thread));
		if(!thread)
		{
			err("out of memory");
			exit(EXIT_FAILURE);
		}

		thread->tid = tid;
		thread->cpu = conv->ncpus + conv->nvcpus++;
		thread->external = thread->cpu;
		thread->state = THREAD_ST_UNKNOWN;
		thread->task = NULL;
		thread->busy_wait = 0;
		thread->color = conv->last_thread_color++;
		thread->n_stack = 0;

		HASH_ADD(hh, conv->threads, tid, sizeof(thread->tid), thread);
		dbg("new thread %p\n", thread);
	}
	else
	{
		dbg("reusing thread %p\n", thread);
	}

	assert(thread->external);
	assert(thread->tid == tid);
	assert(thread->cpu >= conv->ncpus);

	conv->cpus[thread->cpu].thread = thread;
	dbg("thread at cpu %d is %p\n", thread->cpu,
			conv->cpus[thread->cpu].thread);

	return thread->cpu;
}

int64_t
get_env_int64(const bt_trace *trace, const char *name)
{
	const bt_value *value;

	/* Find a value using the name */
	value = bt_trace_borrow_environment_entry_value_by_name_const(
			trace, name);

	assert(value);

	return bt_value_integer_signed_get(value);
}

static void
parse_metadata(struct conv *conv, const bt_message *message)
{
	const bt_event *event;
	const bt_packet *packet;
	const bt_stream *stream;
	const bt_trace *trace;
	const bt_value *value;
	const char *cpulist;
	int64_t raw_start, raw_end;

	event = bt_message_event_borrow_event_const(message);
	assert(event);
	packet = bt_event_borrow_packet_const(event);
	assert(packet);
	stream = bt_packet_borrow_stream_const(packet);
	assert(packet);
	stream = bt_packet_borrow_stream_const(packet);
	assert(stream);
	trace = bt_stream_borrow_trace_const(stream);
	assert(trace);

	value = bt_trace_borrow_environment_entry_value_by_name_const(
			trace, "cpu_list");
	assert(value);

	assert(bt_value_is_string(value));

	cpulist = bt_value_string_get(value);

	/* Physical CPUs */
	populate_cpus(conv, cpulist);

	/* Begin without virtual CPUs; they are populated as external
	 * threads are created (including the leader thread) */
	conv->nvcpus = 0;

	/* Get the clock offset using the time_correction field */
	conv->clock_offset_ns = get_env_int64(trace, "time_correction");

	/* Get the start and end timestamps */
	raw_start = get_env_int64(trace, "start_ts");
	raw_end = get_env_int64(trace, "end_ts");

	/* The compute the *actual* values for the current trace by
	 * using the clock offset. An event time value must be between
	 * start_ns and end_ns */
	conv->clock_start_ns = conv->clock_offset_ns;
	conv->clock_end_ns = raw_end - raw_start +
		conv->clock_offset_ns;

	conv->external_threads = get_env_int64(trace, "external_thread_count");
}

/*
* Prints a line for `message`, if it's an event message, to the
* standard output.
*/
static void
print_message(struct conv *conv, const bt_message *message)
{
	uint32_t msg_type = bt_message_get_type(message);

	/* Discard if it's not an event message */
	if (msg_type != BT_MESSAGE_TYPE_EVENT)
	{
		return;
	}

	/* FIXME: This call must be done when the stream begins to avoid
	 * the "if" */
	if(conv->ncpus < 0)
		parse_metadata(conv, message);

	if(conv->ncpus < 0)
	{
		err("bad ncpus\n");
		exit(EXIT_FAILURE);
	}

	/* Borrow the event message's event and its class */
	const bt_event *event = bt_message_event_borrow_event_const(message);
	const bt_event_class *event_class = bt_event_borrow_class_const(event);
	uint64_t class_id = bt_event_class_get_id(event_class);

	const bt_packet *packet = bt_event_borrow_packet_const(event);

	assert(packet);

	const bt_field *packet_field =
		bt_packet_borrow_context_field_const(packet);
	const bt_field *payload_field =
		bt_event_borrow_payload_field_const(event);

	assert(packet_field);
	assert(payload_field);

	const bt_field* cpu_field =
		bt_field_structure_borrow_member_field_by_name_const(
				packet_field, "cpu_id");

	assert(cpu_field);

	uint64_t cpu_id =
		bt_field_integer_unsigned_get_value(cpu_field);

	dbg("--- got event cpu_id=%lu class_id=%lu\n", cpu_id, class_id);

	/* The cpu_id from the event must the handled differently if
	 * it's a physical or virtual CPU. We can use the max_pcpu to
	 * distinguish the case, as virtual cpus are always larger than
	 * the largest physical CPU */

	assert(cpu_id >= 0);
	if(cpu_id > conv->max_pcpu)
	{
		/* Virtual CPU */
		conv->curr_cpu = get_external_thread_cpu(conv, event);
		assert(conv->curr_cpu >= conv->ncpus);
	}
	else
	{
		/* Physical CPU */
		conv->curr_cpu = conv->pcpu_index[cpu_id];
		assert(conv->curr_cpu >= 0);
		assert(conv->curr_cpu < conv->ncpus);
	}

	int i, ignored;

	for(i=0, ignored=1; i<MAX_HOOKS; i++)
	{
		if(conv->hook_table[class_id][i] == NULL)
			break;

		conv->hook_table[class_id][i](conv, event, class_id);
		ignored = 0;
	}

	if(ignored)
		conv->ev_ignored++;

	conv->ev_processed++;

	if(conv->n_acc_ev)
		flush_acc_events(conv, message);

	conv->curr_cpu = -1;

	if(get_time() - conv->last_reported > REPORT_TIME)
	{
		double dt;

		dt = get_time() - conv->tic;

		size_t written = ftell(conv->prv);
		double speed = (double) (written - conv->last_written) / dt;

		if(!conv->quiet)
		{
			fprintf(stderr, "\r%ld MB (%d%%) written at %.2f MB/s",
				written / (1024 * 1024),
				(int) (conv->ev_progress * 100.0),
				speed / (1024 * 1024));
		}

		conv->tic = get_time();
		conv->ev_last = conv->ev_processed;
		conv->last_written = written;
		conv->last_reported = conv->tic;
	}
}

/*
 * Consumes a batch of messages and writes the corresponding lines to
 * the standard output.
 */
bt_component_class_sink_consume_method_status conv_consume(
	bt_self_component_sink *self_component_sink)
{
	bt_component_class_sink_consume_method_status status =
	BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_OK;

	/* Retrieve our private data from the component's user data */
	struct conv *conv = bt_self_component_get_data(
	bt_self_component_sink_as_self_component(self_component_sink));

	/* Consume a batch of messages from the upstream message iterator */
	bt_message_array_const messages;
	uint64_t message_count;
	bt_message_iterator_next_status next_status =
	bt_message_iterator_next(conv->message_iterator, &messages,
		&message_count);

	switch (next_status) {
		case BT_MESSAGE_ITERATOR_NEXT_STATUS_END:
		/* End of iteration: put the message iterator's reference */
		bt_message_iterator_put_ref(conv->message_iterator);
		status = BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_END;
		goto end;
		case BT_MESSAGE_ITERATOR_NEXT_STATUS_AGAIN:
		status = BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_AGAIN;
		goto end;
		case BT_MESSAGE_ITERATOR_NEXT_STATUS_MEMORY_ERROR:
		status = BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_MEMORY_ERROR;
		goto end;
		case BT_MESSAGE_ITERATOR_NEXT_STATUS_ERROR:
		status = BT_COMPONENT_CLASS_SINK_CONSUME_METHOD_STATUS_ERROR;
		goto end;
		default:
		break;
	}

	/* For each consumed message */
	for (uint64_t i = 0; i < message_count; i++) {
		/* Current message */
		const bt_message *message = messages[i];

		/* Print line for current message if it's an event message */
		print_message(conv, message);

		/* Put this message's reference */
		bt_message_put_ref(message);
	}

end:
	return status;
}

/* Mandatory */
BT_PLUGIN_MODULE();

/* Define the `prv` plugin */
BT_PLUGIN(ctfast);

/* Define the `output` sink component class */
BT_PLUGIN_SINK_COMPONENT_CLASS(prv, conv_consume);

/* Set some of the `output` sink component class's optional methods */
BT_PLUGIN_SINK_COMPONENT_CLASS_INITIALIZE_METHOD(prv,
	conv_initialize);
BT_PLUGIN_SINK_COMPONENT_CLASS_FINALIZE_METHOD(prv, conv_finalize);
BT_PLUGIN_SINK_COMPONENT_CLASS_GRAPH_IS_CONFIGURED_METHOD(prv,
	conv_graph_is_configured);
