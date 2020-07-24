/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <atomic>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <sys/mman.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <dlfcn.h>

#include <algorithm> // TODO remove me

#include "CTFKernelEventsProvider.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#include <iostream> // TODO remove me


// TODO remove me
#define TP_SCHED_SWITCH 318
#define TP_SCHED_WAKEUP 320


#define PAGE_SIZE (1<<12)

// TODO remoeve me
#define COUNT_OF(x) (sizeof(x)/sizeof(0[x]))

struct __attribute__((__packed__)) eventHeader {
	ctf_kernel_event_id_t id;
	uint64_t timestamp;
};

struct __attribute__((__packed__)) tp_common {
	unsigned short common_type;
	unsigned char  common_flags;
	unsigned char  common_preempt_count;
	int            common_pid;
};

struct perf_record_sample {
	struct perf_event_header header;
	uint64_t time;   // if PERF_SAMPLE_TIME
	uint32_t size;   // if PERF_SAMPLE_RAW
	char     data[]; // if PERF_SAMPLE_RAW
};

struct perf_lost_record {
	struct perf_event_header header;
	uint64_t id;
	uint64_t lost;
	// struct sample_id sample_id;
};

// TODO DELETEME
struct __attribute__((__packed__)) tp_sched_switch {
	struct tp_common tpc;
	char  prev_comm[16];
	pid_t prev_pid;
	int   prev_prio;
	long  prev_state;
	char  next_comm[16];
	pid_t next_pid;
	int   next_prio;
};

// TODO DELETEME
struct __attribute__((__packed__)) tp_sched_wakeup {
	struct tp_common tpc;
	char  comm[16];
	pid_t pid;
	int   prio;
	int   success;
	int   target_cpu;
};

std::vector<ctf_kernel_event_id_t>   * CTFAPI::CTFKernelEventsProvider::_enabledEvents;
std::vector<ctf_kernel_event_size_t> * CTFAPI::CTFKernelEventsProvider::_eventSizes;

uint64_t CTFAPI::CTFKernelEventsProvider::_referenceTimestamp = 0;


// atomic functions borrowed from libiouring:include/liburing/barrier.h
template <typename T>
static inline void smp_store_release(T *p, T v)
{
	std::atomic_store_explicit(reinterpret_cast<std::atomic<T> *>(p), v,
				   std::memory_order_release);
}

template <typename T>
static inline T smp_load_acquire(const T *p)
{
	return std::atomic_load_explicit(
		reinterpret_cast<const std::atomic<T> *>(p),
		std::memory_order_acquire);
}

// TODO DELETEME
void check_record_sample_size(struct perf_record_sample *prs)
{
	size_t total, record_extra;
	size_t unknown;

	total = prs->header.size;
	record_extra = sizeof(uint64_t) /*time*/ + sizeof(uint32_t); /*size*/

	unknown = total - sizeof(struct perf_event_header)
			- record_extra
			- prs->size;

	if (unknown) {
		fprintf(stderr,
			"WARNING: sample record unknown bytes: %zu\n"
			" header.size  = %u\n"
			" size         = %u\n"
			" record extra = %zu\n",
			unknown, prs->header.size, prs->size, record_extra);
		//perf_print_header_type(&prs->header);
		//fprintf(stderr, "event count=%" PRIu64 " bytes=%lu pages=%f\n", cnt,
		//	cnt*prs->header.size,
		//	(float)(cnt*prs->header.size)/4096.0);
	}
}

// TODO DELETEME
void check_tp_record_size(struct perf_record_sample *prs, const char *name, size_t expected_fields_size)
{
	ssize_t record_extra;
	ssize_t unknown;
	unsigned char align;
	unsigned char padding;

	record_extra = sizeof(uint64_t) /*time*/ + sizeof(uint32_t); /*size*/

	// data is 64 bit aligned, padded with 0
	align = ((((uintptr_t) prs->data) + expected_fields_size) & 0x7UL);
	padding = align ? 8 - align : 0;

	// The data returned in prv->data should only contain the common
	// tracepoint fields, the expected fields for this specific tracepoint,
	// and padding. The expected_fields_size accounts for the both common
	// and specific tracepoint fields.
	unknown = prs->size - expected_fields_size
			    - padding;

	if (unknown) {
		fprintf(stderr,
			"WARNING: tracepoint %s unknown bytes: %zd\n"
			" header.size             = %u\n"
			" sizeof(header)          = %zd\n"
			" record extra            = %zd\n"
			" data size               = %u\n"
			" expected tp fields size = %zu\n"
			" padding                 = %u\n"
			" record addr             = %p\n"
			" data addr               = %p\n",
			name, unknown,
			prs->header.size, sizeof(struct perf_event_header), record_extra,
			prs->size, expected_fields_size, padding,
			prs, prs->data);
	}
}

// TODO deleteme
void print_tracepoint(struct perf_record_sample *prs)
{
	struct tp_common *tpc;


	//int i;
	//printf("raw dump\n");
	//for (i = 0; i < prs->header.size; i++) {
	//	printf("%c",((char *)prs)[i]);
	//}
	//printf("\nend raw dump\n");


	printf("%" PRIu64 " ", prs->time);

	tpc = (struct tp_common *) prs->data;
	printf("{common_type = %d, common_flags=%u, common_preempt_count=%u, common_pid=%d} ",
	       tpc->common_type, tpc->common_flags, tpc->common_preempt_count, tpc->common_pid);


	switch(tpc->common_type) {
	case TP_SCHED_SWITCH:
	{
		struct tp_sched_switch *tp;
		tp = (struct tp_sched_switch *) tpc;
		check_tp_record_size(prs, "sched_switch", sizeof(*tp));
		printf("{prev_comm=%.16s, prev_pid=%d, prev_prio=%d, prev_state=%ld,"
		       " next_comm=%.16s, next_pid=%d, next_prio=%d}\n",
		       tp->prev_comm, tp->prev_pid, tp->prev_prio, tp->prev_state,
		       tp->next_comm, tp->next_pid, tp->next_prio);
		break;
	}
	case TP_SCHED_WAKEUP:
	{
		struct tp_sched_wakeup *tp;
		tp = (struct tp_sched_wakeup *) tpc;
		check_tp_record_size(prs, "sched_wakeup", sizeof(*tp));
		printf("{comm=%.16s, pid=%d, prio=%d, success=%d target_cpu=%d}\n",
		       tp->comm, tp->pid, tp->prio, tp->success, tp->target_cpu);
		break;
	}
	default:
		fprintf(stderr, "Error: Unknown tracepoint type: %d\n",
			tpc->common_type);
		exit(EXIT_FAILURE);
	}
}

// TODO DELETEME ?
void report_lost_records(struct perf_lost_record *plr)
{
	fprintf(stderr, "\n\nrecords lost: id=%" PRIu64 " lost=%" PRIu64 "\n\n",
		plr->id, plr->lost);

}

#define PERF_HEADER_ENTRY(name)  \
case name: \
	printf(#name " "); \
	break;

void perf_print_header_type(struct perf_event_header *header)
{
	printf(" - entry: ");
	switch (header->type) {
	PERF_HEADER_ENTRY(PERF_RECORD_MMAP)
	PERF_HEADER_ENTRY(PERF_RECORD_LOST)
	PERF_HEADER_ENTRY(PERF_RECORD_COMM)
	PERF_HEADER_ENTRY(PERF_RECORD_EXIT)
	PERF_HEADER_ENTRY(PERF_RECORD_THROTTLE)
	PERF_HEADER_ENTRY(PERF_RECORD_UNTHROTTLE)
	PERF_HEADER_ENTRY(PERF_RECORD_FORK)
	PERF_HEADER_ENTRY(PERF_RECORD_READ)
	PERF_HEADER_ENTRY(PERF_RECORD_SAMPLE)
	default:
		printf("[type %d not known]", header->type);
	}

	switch (header->misc & PERF_RECORD_MISC_CPUMODE_MASK) {
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_CPUMODE_UNKNOWN)
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_KERNEL)
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_USER)
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_HYPERVISOR)
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_GUEST_KERNEL)
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_GUEST_USER)
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_MMAP_DATA)
	default:
		printf("[misc %d not known]", header->misc);
	}

	if (header->misc & PERF_RECORD_MISC_MMAP_DATA)
		printf("PERF_RECORD_MISC_MMAP_DATA");
	else if (header->misc & PERF_RECORD_MISC_EXACT_IP)
		printf("PERF_RECORD_MISC_EXACT_IP");
	else if (header->misc & PERF_RECORD_MISC_EXT_RESERVED)
		printf("PERF_RECORD_MISC_EXT_RESERVED");

	printf("\n");
	printf(" - header size: %" PRIu16 "\n", header->size);
}


long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags)
{
	int ret;
	ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
		      group_fd, flags);
	return ret;
}

CTFAPI::CTFKernelEventsProvider::CTFKernelEventsProvider(int cpu, size_t userSize)
	:
	_lastTimestamp(0),
	_lastCurrent(0),
	_lastId(0),
	_eventHeaderSize(8 + 2), // timestamnp + id
	_lost(0),
	_throttle(0),
	_unthrottle(0),
	_numberOfEvents(0),
	_numberOfUnorderedEvents(0),
	_kernelBuffer(nullptr)
{
	struct perf_event_attr pe;
	pid_t pid;
	unsigned long flags;
	uint64_t alignedSize;
	void *perfMap;

	alignedSize  = (userSize + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
	_metaSize    = PAGE_SIZE;
	_dataSize    = alignedSize;
	_totalSize   = _metaSize + _dataSize;
	_dataMask    = _dataSize - 1;
	_tempSize    = (1 << 16); // perf sample has 16 bits size limit
	// TODO temporal buffer is quite big. Cannot do something else?

	// Initialize common perf attribute argument
	memset(&pe, 0, sizeof(struct perf_event_attr));
	pe.type = PERF_TYPE_TRACEPOINT;
	pe.size = sizeof(struct perf_event_attr);
	pe.sample_period = 1; // record every n-events, setting to 1 records all
	pe.sample_type = PERF_SAMPLE_TIME | PERF_SAMPLE_RAW;
	pe.wakeup_events = 9999999;
	pe.disabled = 1;
	pe.use_clockid = 1;
	pe.clockid = CLOCK_MONOTONIC_RAW;

	// tracing other than the current thread requires CAP_SYS_ADMIN
	// capability of /proc/sys/kernel/perf_event_paranoid value less than 1.
	pid = -1;
#ifdef PERF_FLAG_FD_CLOEXEC
	flags = PERF_FLAG_FD_CLOEXEC;
#else
	flags = 0;
#endif

	// open events
	_eventsFds.resize(_enabledEvents->size());
	for (long unsigned int i = 0; i < _eventsFds.size(); i++) {
		pe.config = (*_enabledEvents)[i];
		_eventsFds[i] = perf_event_open(&pe, pid, cpu, -1, flags);
		if (_eventsFds[i] == -1) {
			FatalErrorHandler::fail(
				"CTF: Kernel: When calling perf_event_open: ",
				strerror(errno)
			);
		}
	}

	perfMap = mmap(NULL, _totalSize,
		       PROT_READ | PROT_WRITE, MAP_SHARED,
		       _eventsFds[0], 0);
	if (perfMap == MAP_FAILED) {
		FatalErrorHandler::fail(
			"CTF: Kernel: When mapping perf pages: ",
			strerror(errno)
		);
	}

	if (fcntl(_eventsFds[0], F_SETFL, O_RDONLY | O_NONBLOCK)) {
		FatalErrorHandler::fail(
			"CTF: Kernel: When changing file desciptor properties: ",
			strerror(errno)
		);
	}

	// Redirect secondary events output to primary event buffer
	// This cannot be done before mmap!
	for (long unsigned int i = 1; i < _eventsFds.size(); i++) {
		if (ioctl(_eventsFds[i], PERF_EVENT_IOC_SET_OUTPUT, _eventsFds[0]) == -1) {
			FatalErrorHandler::fail(
				"CTF: Kernel: redirecting event file descriptor: ",
				strerror(errno)
			);
		}
		if (fcntl(_eventsFds[i], F_SETFL, O_RDONLY | O_NONBLOCK)) {
			FatalErrorHandler::fail(
				"CTF: Kernel: changing file desciptor properties: ",
				strerror(errno)
			);
		}
	}

	_temporalBuffer = malloc(_tempSize);
	if (!_temporalBuffer) {
		FatalErrorHandler::fail(
			"CTF: Kernel: Cannot allocate memory for temporal buffer: ",
			strerror(errno)
		);
	}

	_kernelBuffer = perfMap;
	_dataRing     = ((char *)perfMap) + PAGE_SIZE;
	_metaPage     = _kernelBuffer;

	// get ring buffer initial positions
	struct perf_event_mmap_page *metaPage = (struct perf_event_mmap_page *) _metaPage;

	//TODO I think barriers are not needed at this point because we have not
	//started tracing yet
	_localHead = smp_load_acquire(&metaPage->data_head);
	_localTail = smp_load_acquire(&metaPage->data_tail);

	_current = _localTail;

	// Not all perf versions have the data_offset field. Disabling assert for now
	assert(metaPage->data_offset == PAGE_SIZE);
	assert(metaPage->data_size == _dataSize);
}

void *CTFAPI::CTFKernelEventsProvider::getNextEvent(uint64_t current)
{
	struct perf_event_header *header;

	// The record might be split between the beginning and the end
	// of the perf buffer.
	//
	// It's not possible to create a "magic ring buffer" as mmaping
	// the perf buffer with an offset is not allowed. Therefore the
	// only way is to copy the struture in an intermediate buffer
	//
	// The header cannot be split because it's 64 bit aligned and 64
	// bit long. Hence, it's safe to access it.
	header = (struct perf_event_header *) (_dataRing + (current & _dataMask));

	// adapted from perf tools/perf/util/mmap.c:perf_mmap__read()
	if (((current & _dataMask) + header->size) != ((current + header->size) & _dataMask)) {
		//std::cout << "!!!!!!!!!!!!!!!!! buffer wrap !!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
		uint64_t offset = current;
		uint64_t len = header->size;
		uint64_t cpy;
		char *dst = (char *) _temporalBuffer;

		do {
			cpy = std::min(_dataSize - (offset & _dataMask), len);
			memcpy(dst, _dataRing + (offset & _dataMask), cpy);
			offset += cpy;
			dst += cpy;
			len -= cpy;
		} while (len);

		// TODO where the fork is union perf_event defined?
		//event = (union perf_event *)map->event_copy;
		header = (struct perf_event_header *) _temporalBuffer;
	}

	return header;
}

bool CTFAPI::CTFKernelEventsProvider::read(void *buf, uint64_t size, uint64_t *read)
{
	struct perf_event_header *header;
	struct perf_record_sample *prs;
	struct perf_lost_record *plr;
	struct tp_common *eventCommonFields;
	struct eventHeader *eventHeader;

	ctf_kernel_event_id_t eventId;
	ctf_kernel_event_id_t payloadSize;
	uint64_t tracepointSize = 0;
	uint64_t processedBytes = 0;
	uint64_t remainingBytes = size;
	char *buffer = (char *) buf;

	while (_current != _localHead) {

		header = (struct perf_event_header *) getNextEvent(_current);

		//std::cout << "current: " << _current << " size: " << header->size << " :: " << (_current & _dataMask) << "/" << _dataSize << std::endl;

		// process record
		switch(header->type) {
		case PERF_RECORD_SAMPLE:
		{
			prs = (struct perf_record_sample *) header;
			eventCommonFields = (struct tp_common *) prs->data;
			eventId = eventCommonFields->common_type;
			payloadSize = (*_eventSizes)[eventId];
			assert(payloadSize <= prs->size - sizeof(struct tp_common));

			check_record_sample_size(prs);
			//print_tracepoint(prs);

			tracepointSize = _eventHeaderSize + payloadSize;
			if (tracepointSize <= remainingBytes) {
				// copy event header
				eventHeader = (struct eventHeader *) buffer;
				eventHeader->id = eventId;
				eventHeader->timestamp = (prs->time - _referenceTimestamp);

				// unordered events checker
				if (prs->time < _lastTimestamp) {
					_numberOfUnorderedEvents++;
					// TODO cleanup needed
					//std::cerr << "!!!!!!!! WARNING:::Unordered events detected:::WARNING !!!!!!!!" << std::endl;
					//std::cerr << " - prev: " << _lastTimestamp << "\thead: " << _lastCurrent << "\tid: " << _lastId << std::endl;
					//std::cerr << " - curr: " << prs->time      << "\thead: " << _current     << "\tid: " << eventId << std::endl;
				}
				_lastTimestamp = prs->time;
				_lastCurrent   = _current;
				_lastId        = eventId;

				// copy event payload
				memcpy(buffer + sizeof(struct eventHeader),
				       prs->data + sizeof(struct tp_common),
				       payloadSize
				);
				// account written tracepoint
				buffer         += tracepointSize;
				remainingBytes -= tracepointSize;
				processedBytes += tracepointSize;
				_numberOfEvents++;
			} else {
				goto early_end;
			}

			break;
		}
		case PERF_RECORD_LOST:
		{
			plr = (struct perf_lost_record *) header;
			report_lost_records(plr);
			_lost += plr->lost;
			break;
		}
		case PERF_RECORD_THROTTLE:
		{
			_throttle++;
			break;
		}
		case PERF_RECORD_UNTHROTTLE:
		{
			_unthrottle++;
			break;
		}
		default:
			perf_print_header_type(header);
			//FatalErrorHandler::fail("CTF: Kernel: Unknown perf record: ",
			//	header->type);
		}

		_current += header->size;
	}

early_end:

	if (processedBytes) {
		// Good ending. We have copied at least one event
		*read = processedBytes;
		return true;
	} else {
		// Bad ending. We have __not__ copied a single event
		// return the size of the event that we couldn't copy or 0 if
		// there was really no tracepoint event
		*read = tracepointSize;
		return false;
	}
}

CTFAPI::CTFKernelEventsProvider::~CTFKernelEventsProvider()
{
	if (munmap(_kernelBuffer, _totalSize)) {
		FatalErrorHandler::warn(
			"CTF: Kernel: When unmapping perf buffer: ",
			strerror(errno)
		);
	}

	free(_temporalBuffer);
}

void CTFAPI::CTFKernelEventsProvider::enable()
{
	// TODO check if all events can be enabled/disabled at once with
	// PERF_IOC_FLAG_GROUP. see man perf_event_open and
	// PERF_EVENT_IOC_ENABLE
	for (long unsigned int i = 0; i < _eventsFds.size(); i++)
		ioctl(_eventsFds[i], PERF_EVENT_IOC_ENABLE, 0);
}

void CTFAPI::CTFKernelEventsProvider::disable()
{
	// TODO check if all events can be enabled/disabled at once with
	// PERF_IOC_FLAG_GROUP. see man perf_event_open and
	// PERF_EVENT_IOC_ENABLE
	for (long unsigned int i = 0; i < _eventsFds.size(); i++)
		ioctl(_eventsFds[i], PERF_EVENT_IOC_DISABLE, 0);
}

bool CTFAPI::CTFKernelEventsProvider::hasEvents()
{
	return _current != _localHead;
}

void CTFAPI::CTFKernelEventsProvider::updateHead()
{
	struct perf_event_mmap_page *metaPage = (struct perf_event_mmap_page *) _metaPage;

	_localHead = smp_load_acquire(&metaPage->data_head);
}

void CTFAPI::CTFKernelEventsProvider::updateTail()
{
	struct perf_event_mmap_page *metaPage = (struct perf_event_mmap_page *) _metaPage;

	_localTail = _current;
	smp_store_release(&metaPage->data_tail, _localTail);
}
