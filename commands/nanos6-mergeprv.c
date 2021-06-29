/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>

#define MAX_RANKS 1024
#define MAX_LINE 1024
#define MAX_CPUS 256
#define MAX_EV 64
#define PROGNAME "nanos6-mergeprv"
#define OUTPRV "trace.prv"
#define OUTPCF "trace.pcf"
#define OUTROW "trace.row"
#define INPCF OUTPCF
#define INROW OUTROW
#define TRACE_VERSION 1

/* The spec says:
 * 2:cpu_id:appl_id:task_id:thread_id:time:event_type:event_value
 *
 * but we use:
 * 2:_:_:_:cpu:time_ns:event_type:event_value
 * */
struct event {
	int cpu;
	int app;
	int rank;
	int thread; /* Used by CTF for the "cpu" */
	int64_t t; /* In nanoseconds (delta) */
	int64_t type;
	int64_t val;
};

struct merger {
	FILE *prv[MAX_RANKS];
	struct event ev[MAX_RANKS];
	int ranks;
	int active[MAX_RANKS];
	int64_t last_time;
	FILE *outprv;
	FILE *outrow;
	int total_threads;
};

struct prv_header {
	int day, month, year;
	int hour, minute;
	int64_t timespan;
	int nodes;
	int cpus[MAX_RANKS];
	int apps;
	int tasks;
	int threads;
	int running_node;
};

int
next_event(FILE *f, struct event *ev)
{
	int ntok, type;

	/* TODO: Support communication and state lines */

	while(1)
	{
		/* 2:0:1:1:2:22810769:6400017:7 */
		/* 2:cpu_id:appl_id:task_id:thread_id:time:event_type:event_value */
		ntok = fscanf(f, "%d:%d:%d:%d:%d:%ld:%ld:%ld",
				&type, &ev->cpu, &ev->app, &ev->rank,
				&ev->thread, &ev->t, &ev->type, &ev->val);

		if(ntok == EOF)
			return -1;

		/* Discard non-events by now */
		if(ntok >= 1 && type != 2)
		{
			fprintf(stderr, "warning: ignoring unsupported type %d\n",
					type);
			continue;
		}

		if(ntok != 8)
		{
			fprintf(stderr, "error: line with %d tokens, instead of 8\n", ntok);
			fprintf(stderr, "Do you use one event per line?\n");
			exit(EXIT_FAILURE);
		}

		return 0;
	}

	/* Not reached */
	return 0;
}


void
usage(int argc, char *argv[])
{
	fprintf(stderr, "%s: merge multi-rank PRV files into one\n",
			PROGNAME);
	fprintf(stderr, "\n");
	fprintf(stderr, "Usage: %s <trace dir>\n", PROGNAME);
	fprintf(stderr, "\n");
	fprintf(stderr, "Trace version: %d\n", TRACE_VERSION);
	exit(EXIT_FAILURE);
}

void
emit(FILE *f, struct event *ev)
{
 	/* 2:cpu_id:appl_id:task_id:thread_id:time:event_type:event_value */
	fprintf(f, "2:%d:%d:%d:%d:%ld:%ld:%ld\n",
			ev->cpu, ev->app, ev->rank, ev->thread,
			ev->t, ev->type, ev->val);
}

void
read_prv_header(FILE *f, struct prv_header *h)
{
	int ntok;

	/* Read the constant field part first */
	ntok = fscanf(f, "#Paraver (%d/%d/%d at %d:%d):%" SCNd64 "_ns:%d:",
			&h->day, &h->month, &h->year,
			&h->hour, &h->minute,
			&h->timespan, &h->nodes);

	if(ntok != 7)
	{
		fprintf(stderr, "bad PRV header (1): expecting 7 tokens, found %d\n",
				ntok);

		exit(EXIT_FAILURE);
	}

	/* The resource model is not used by now, so we can skip the
	 * parsing of the cpus */
	if(h->nodes != 0)
	{
		fprintf(stderr, "resource model not supported\n");
		exit(EXIT_FAILURE);
	}

	ntok = fscanf(f, "%d:%d(", &h->apps, &h->tasks);

	if(ntok != 2)
	{
		fprintf(stderr, "bad PRV header (2): expecting 2 tokens, found %d\n",
				ntok);

		exit(EXIT_FAILURE);
	}

	/* Only one app for now */
	if(h->apps != 1)
	{
		fprintf(stderr, "only one application supported\n");
		exit(EXIT_FAILURE);
	}

	/* The given trace must contain only one task */
	if(h->tasks != 1)
	{
		fprintf(stderr, "only one task supported\n");
		exit(EXIT_FAILURE);
	}

	ntok = fscanf(f, "%d:%d)", &h->threads, &h->running_node);

	if(ntok != 2)
	{
		fprintf(stderr, "bad PRV header (3): expecting 2 tokens, found %d\n",
				ntok);

		exit(EXIT_FAILURE);
	}

	/* The running node is not used, must be 1 */
	if(h->running_node != 1)
	{
		fprintf(stderr, "unexpected running node (%d != 1)\n",
				h->running_node);
		exit(EXIT_FAILURE);
	}
}

void
merge_prv_header(struct merger *m)
{
	char sep;
	int i, running_node, nodes, apps;
	int64_t timespan;
	struct prv_header h;
	int threads[MAX_RANKS];

	m->total_threads = 0;

	/* Read the PRV header */
	for(i=0; i < m->ranks; i++)
	{
		read_prv_header(m->prv[i], &h);

		/* Save the threads for every rank */
		threads[i] = h.threads;

		/* Accumulate the total threads */
		m->total_threads += h.threads;
	}

	/* Don't provide the resource model information */
	nodes = 0;

	/* Only one app */
	apps = 1;
	
	/* Timespan is not used */
	timespan = 0;

	fprintf(m->outprv, "#Paraver (%02d/%02d/%02d at %02d:%02d):%"
			PRIi64 "_ns:%d:%d:%d(",
			h.day, h.month, h.year,
			h.hour, h.minute,
			timespan, nodes, apps, m->ranks);

	/* The running node is always the same */
	running_node = 1;

	for(i=0, sep=','; i < m->ranks; i++)
	{
		/* Last element closes the parenthesis */
		if(i == m->ranks - 1)
			sep = ')';

		fprintf(m->outprv, "%d:%d%c", threads[i], running_node, sep);
	}

	/* Finish the header */
	fprintf(m->outprv, "\n");
}

void
merge_prv_events(struct merger *m)
{
	int i, cur;

	/* Populate first events */
	for(i=0; i < m->ranks; i++)
		if(next_event(m->prv[i], &m->ev[i]) != 0)
			m->active[i] = 0;

	while(1)
	{
		/* Find the next event with the lowest time */
		for(i=0, cur=-1; i < m->ranks; i++)
		{
			if(!m->active[i]) continue;

			if(cur < 0)
				cur = i;

			if(m->ev[i].t < m->ev[cur].t)
				cur = i;
		}

		/* No more events: all ranks depleted */
		if(cur < 0)
			break;

		/* Fix the rank (starting in 1) */
		m->ev[cur].rank = cur + 1;

		/* Emit the event */
		emit(m->outprv, &m->ev[cur]);

		/* Get another event for that rank (if any) */
		if(next_event(m->prv[cur], &m->ev[cur]) != 0)
			m->active[cur] = 0;
	}
}

void
write_pcf(struct merger *m)
{
	char buf[MAX_LINE];
	FILE *in, *out;
	size_t n;

	if((out = fopen(OUTPCF, "w")) == NULL)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	if((in = fopen("0/prv/" INPCF, "r")) == NULL)
	{
		perror("fopen");
		exit(EXIT_FAILURE);
	}

	while((n = fread(buf, 1, MAX_LINE, in)) > 0)
	{
		if(fwrite(buf, 1, n, out) != n)
		{
			fprintf(stderr, "fwrite failed to write %ld bytes\n", n);
			exit(EXIT_FAILURE);
		}
	}

	fclose(in);
	fclose(out);
}

void
write_row(struct merger *m)
{
	int i;
	char buf[MAX_LINE];
	char rowpath[PATH_MAX];
	FILE *row;
	char *marker = "LEVEL THREAD SIZE";
	int markerlen;

	markerlen = strlen(marker);

	fprintf(m->outrow, "LEVEL NODE SIZE %d\n", 1);
	fprintf(m->outrow, "hostname\n");
	fprintf(m->outrow, "\n");
	fprintf(m->outrow, "LEVEL THREAD SIZE %d\n", m->total_threads);

	for(i=0; i<m->ranks; i++)
	{
		snprintf(rowpath, PATH_MAX - 1, "%d/prv/" INROW, i);
		if((row = fopen(rowpath, "r")) == NULL)
		{
			fprintf(stderr, "cannot open input ROW file %s: %s\n",
					rowpath, strerror(errno));
			exit(EXIT_FAILURE);
		}

		/* Find the LEVEL THREAD line */
		while(fgets(buf, MAX_LINE, row) != NULL)
		{
			buf[markerlen] = '\0';
			if(strcmp(buf, marker) == 0)
				break;
		}

		/* Then write the row names with the prefix */
		while(fgets(buf, MAX_LINE, row) != NULL)
			fprintf(m->outrow, "RANK %d %s", i, buf);

		fclose(row);
	}
}

void
merge_prv(struct merger *m)
{
	merge_prv_header(m);
	merge_prv_events(m);

	write_pcf(m);
	write_row(m);
}

void
check_version()
{
	FILE *f;
	int ver;

	if((f = fopen("VERSION", "r")) == NULL)
	{
		perror("cannot open VERSION file");
		exit(EXIT_FAILURE);
	}

	if(fscanf(f, "%d", &ver) != 1)
	{
		fprintf(stderr, "failed to read version number\n");
		exit(EXIT_FAILURE);
	}

	if(ver != TRACE_VERSION)
	{
		fprintf(stderr, "unsupported trace version: %d (instead of %d)\n",
				ver, TRACE_VERSION);
		exit(EXIT_FAILURE);
	}

	fclose(f);
}

int main(int argc, char *argv[])
{
	struct merger merger;
	char prvpath[PATH_MAX];
	struct stat statbuf;
	int i;

	memset(&merger, 0, sizeof(merger));

	if(argc != 2) usage(argc, argv);

	if(stat(argv[1], &statbuf) != 0)
	{
		fprintf(stderr, "cannot stat trace directory %s: %s\n",
				argv[1], strerror(errno));

		usage(argc, argv);
	}

	if(!S_ISDIR(statbuf.st_mode))
	{
		fprintf(stderr, "the specified path is not a directory: %s\n",
				argv[1]);

		usage(argc, argv);
	}

	if(chdir(argv[1]) != 0)
	{
		perror("chdir");
		exit(EXIT_FAILURE);
	}

	check_version();

	if((merger.outprv = fopen(OUTPRV, "w")) == NULL)
	{
		fprintf(stderr, "cannot open output PRV file %s: %s\n",
				OUTPRV, strerror(errno));
		exit(EXIT_FAILURE);
	}

	if((merger.outrow = fopen(OUTROW, "w")) == NULL)
	{
		fprintf(stderr, "cannot open output ROW file %s: %s\n",
				OUTROW, strerror(errno));
		exit(EXIT_FAILURE);
	}

	/* Look for $rank/prv/trace.prv files */
	for(i=0; i<MAX_RANKS; i++)
	{
		snprintf(prvpath, PATH_MAX, "%d/prv/trace.prv", i);
		if((merger.prv[i] = fopen(prvpath, "r")) == NULL)
		{
			/* If not found, stop here */
			if(errno == ENOENT)
			{
				merger.ranks = i;
				break;
			}

			/* Otherwise abort */
			fprintf(stderr, "cannot open PRV file %s: %s\n",
					prvpath, strerror(errno));
			exit(EXIT_FAILURE);
		}
	}

	if(merger.ranks <= 1)
	{
		fprintf(stderr, "not enough ranks %d\n", merger.ranks);
		exit(EXIT_FAILURE);
	}

	for(i=0; i<merger.ranks; i++)
		merger.active[i] = 1;

	merge_prv(&merger);

	for(i=0; i<merger.ranks; i++)
		fclose(merger.prv[i]);

	fclose(merger.outrow);
	fclose(merger.outprv);

	return 0;
}
