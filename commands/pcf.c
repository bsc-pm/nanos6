/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#include "pcf.h"
#include "cn6.h"

#include <stdio.h>
#include <stdint.h>

const char *pcf_def_header =
	"DEFAULT_OPTIONS\n"
	"\n"
	"LEVEL               THREAD\n"
	"UNITS               NANOSEC\n"
	"LOOK_BACK           100\n"
	"SPEED               1\n"
	"FLAG_ICONS          ENABLED\n"
	"NUM_OF_STATE_COLORS 1000\n"
	"YMAX_SCALE          37\n"
	"\n"
	"\n"
	"DEFAULT_SEMANTIC\n"
	"\n"
	"THREAD_FUNC         State As Is\n";

#define RGB(r, g, b) (r<<16 | g<<8 | b)
#define ARRAY_LEN(x)  (sizeof(x) / sizeof((x)[0]))

/* Define colors for the trace */
#define DEEPBLUE  RGB(  0,   0, 255)
#define LIGHTGREY RGB(217, 217, 217)
#define RED       RGB(230,  25,  75)
#define GREEN     RGB(60,  180,  75)
#define YELLOW    RGB(255, 225,  25)
#define ORANGE    RGB(245, 130,  48)
#define PURPLE    RGB(145,  30, 180)
#define CYAN      RGB( 70, 240, 240)
#define MAGENTA   RGB(240, 50,  230)
#define LIME      RGB(210, 245,  60)
#define PINK      RGB(250, 190, 212)
#define TEAL      RGB(  0, 128, 128)
#define LAVENDER  RGB(220, 190, 255)
#define BROWN     RGB(170, 110,  40)
#define BEIGE     RGB(255, 250, 200)
#define MAROON    RGB(128,   0,   0)
#define MINT      RGB(170, 255, 195)
#define OLIVE     RGB(128, 128,   0)
#define APRICOT   RGB(255, 215, 180)
#define NAVY      RGB(  0,   0, 128)
#define BLUE      RGB(  0, 130, 200)
#define GREY      RGB(128, 128, 128)
#define BLACK     RGB(  0,   0,   0)

const uint32_t pcf_def_palette[] = {
	BLACK,		/* (never shown anyways) */
	BLUE,		/* runtime */
	LIGHTGREY,	/* busy wait */
	RED,		/* task */
	GREEN,
	YELLOW,
	ORANGE,
	PURPLE,
	CYAN,
	MAGENTA,
	LIME,
	PINK,
	TEAL,
	GREY,
	LAVENDER,
	BROWN,
	BEIGE,
	MAROON,
	MINT,
	OLIVE,
	APRICOT,
	NAVY,
	DEEPBLUE
};

const uint32_t *pcf_palette = pcf_def_palette;
const int pcf_palette_len = ARRAY_LEN(pcf_def_palette);

struct event_value {
	int value;
	const char *label;
};

struct event_type {
	int index;
	int type;
	const char *label;
	struct event_value *values;
};

struct event_value runtime_activity_values[] = {
	{ RA_END,	"End" },
	{ RA_RUNTIME,	"Runtime" },
	{ -1, NULL },
};

struct event_type runtime_activity = {
	0, EV_TYPE_RUNTIME_CODE, "Runtime: Runtime Code",
	runtime_activity_values
};

struct event_value runtime_busywaiting_values[] = {
	{ RA_END,		"End" },
	{ RA_BUSYWAITING,	"Busy waiting" },
	{ -1, NULL },
};

struct event_type runtime_busywaiting = {
	0, EV_TYPE_RUNTIME_BUSYWAITING, "Runtime: Busy Waiting",
	runtime_busywaiting_values
};

struct event_value runtime_task_values[] = {
	{ RA_END,	"End" },
	{ RA_TASK,	"Task" },
	{ -1, NULL },
};

struct event_type runtime_task = {
	0, EV_TYPE_RUNTIME_TASKS, "Runtime: Task Code",
	runtime_task_values
};

struct event_value runtime_mode_values[] = {
	{ RM_DEAD,	"Dead" },
	{ RM_RUNTIME,	"Runtime" },
	{ RM_TASK,	"Task" },
	{ -1, NULL },
};

struct event_type runtime_mode = {
	0, EV_TYPE_RUNTIME_MODE, "Runtime: Mode",
	runtime_mode_values
};

struct event_value runtime_subsystems_values[] = {
	{ RS_IDLE,			"Idle" },
	{ RS_RUNTIME,			"Runtime" },
	{ RS_BUSY_WAIT,			"Busy Wait" },
	{ RS_TASK,			"Task" },
	{ RS_DEPENDENCY_REGISTER,	"Dependency: Register" },
	{ RS_DEPENDENCY_UNREGISTER,	"Dependency: Unregister" },
	{ RS_SCHEDULER_ADD_TASK,	"Scheduler: Add Ready Task" },
	{ RS_SCHEDULER_GET_TASK,	"Scheduler: Get Ready Task" },
	{ RS_TASK_CREATE,		"Task: Create" },
	{ RS_TASK_ARGS_INIT,		"Task: Arguments Init" },
	{ RS_TASK_SUBMIT,		"Task: Submit" },
	{ RS_TASKFOR_INIT,		"Task: Taskfor Collaborator Init" },
	{ RS_TASK_WAIT,			"Task: TaskWait" },
	{ RS_WAIT_FOR,			"Task: WaitFor" },
	{ RS_LOCK,			"Task: User Mutex: Lock" },
	{ RS_UNLOCK,			"Task: User Mutex: Unlock" },
	{ RS_BLOCKING_API_BLOCK,	"Task: Blocking API: Block" },
	{ RS_BLOCKING_API_UNBLOCK,	"Task: Blocking API: Unblock" },
	{ RS_SPAWN_FUNCTION,		"SpawnFunction: Spawn" },
	{ RS_SCHEDULER_LOCK_ENTER,	"Scheduler: Lock: Enter" },
	{ RS_SCHEDULER_LOCK_SERVING,	"Scheduler: Lock: Serving tasks" },
	{ -1, NULL },
};

struct event_type runtime_subsystems = {
	0, EV_TYPE_RUNTIME_SUBSYSTEMS, "Runtime Subsystems",
	runtime_subsystems_values
};

struct event_value ctf_flush_values[] = {
	{ 0, "End" },
	{ 1, "Flush" },
	{ -1, NULL },
};

struct event_type ctf_flush = {
	0, EV_TYPE_CTF_FLUSH, "Nanos6 CTF buffers writes to disk",
	ctf_flush_values
};

static void
decompose_rgb(uint32_t col, uint8_t *r, uint8_t *g, uint8_t *b)
{
	*r = (col>>16) & 0xff;
	*g = (col>>8) & 0xff;
	*b = (col>>0) & 0xff;
}

static void
write_header(FILE *f)
{
	fprintf(f, "%s", pcf_def_header);
}

static void
write_colors(FILE *f, const uint32_t *palette, int n)
{
	int i;
	uint32_t col;
	uint8_t r, g, b;

	fprintf(f, "\n\n");
	fprintf(f, "STATES_COLOR\n");

	for(i=0; i<n; i++)
	{
		col = palette[i];
		decompose_rgb(palette[i], &r, &g, &b);
		fprintf(f, "%-3d {%3d, %3d, %3d}\n", i, r, g, b);
	}
}

static void
write_event_type_header(FILE *f, int index, int type, const char *label)
{
	fprintf(f, "\n\n");
	fprintf(f, "EVENT_TYPE\n");
	fprintf(f, "%-4d %-10d %s\n", index, type, label);
}

static void
write_event_type(FILE *f, struct event_type *ev)
{
	int i;

	write_event_type_header(f, ev->index, ev->type, ev->label);

	fprintf(f, "VALUES\n");

	for(i=0; ev->values[i].label; i++)
	{
		fprintf(f, "%-4d %s\n",
				ev->values[i].value,
				ev->values[i].label);
	}
}

static void
write_task_types(struct pcf *pcf, FILE *f)
{
	struct task_type *tt;

	/* Label */
	write_event_type_header(f, 0,
			EV_TYPE_RUNNING_TASK_LABEL,
			"Running Task: Label");

	fprintf(f, "VALUES\n");
	fprintf(f, "%-4d %s\n", 0, "NULL");
	for(tt=pcf->task_types; tt != NULL; tt=tt->hh.next)
		fprintf(f, "%-4lu %s\n", tt->type, tt->label);

	/* Source line */
	write_event_type_header(f, 0,
			EV_TYPE_RUNNING_TASK_SOURCE,
			"Running Task: Source line");

	fprintf(f, "VALUES\n");
	fprintf(f, "%-4d %s\n", 0, "NULL");
	for(tt=pcf->task_types; tt != NULL; tt=tt->hh.next)
		fprintf(f, "%-4lu %s\n", tt->type, tt->srcline);
}

static void
write_events(struct pcf *pcf, FILE *f)
{
	write_event_type(f, &runtime_activity);
	write_event_type(f, &runtime_busywaiting);
	write_event_type(f, &runtime_task);
	write_event_type(f, &runtime_mode);
	write_event_type(f, &runtime_subsystems);
	write_event_type(f, &ctf_flush);

	write_task_types(pcf, f);
}

void
pcf_init(struct pcf *pcf)
{
	pcf->task_types = NULL;
}

int
pcf_write(struct pcf *pcf, FILE *f)
{
	write_header(f);
	write_colors(f, pcf_palette, pcf_palette_len);
	write_events(pcf, f);

	return 0;
}

void
pcf_set_task_types(struct pcf *pcf, struct task_type *task_types)
{
	pcf->task_types = task_types;
}
