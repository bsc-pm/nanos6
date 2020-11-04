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

#include "CTFKernelEventsProvider.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


#define PAGE_SIZE (1<<12)

std::vector<ctf_kernel_event_id_t>   * CTFAPI::CTFKernelEventsProvider::_enabledEvents;
std::vector<ctf_kernel_event_size_t> * CTFAPI::CTFKernelEventsProvider::_eventSizes;

uint64_t CTFAPI::CTFKernelEventsProvider::_referenceTimestamp = 0;

// atomic functions borrowed from libiouring:include/liburing/barrier.h
template <typename T>
static inline void smpStoreRelease(T *p, T v)
{
	std::atomic_store_explicit(reinterpret_cast<std::atomic<T> *>(p), v, std::memory_order_release);
}

template <typename T>
static inline T smpLoadAcquire(const T *p)
{
	return std::atomic_load_explicit(reinterpret_cast<const std::atomic<T> *>(p), std::memory_order_acquire);
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
	_eventHeaderSize(8 + 2), // timestamnp + id
	_lost(0),
	_throttle(0),
	_unthrottle(0),
	_numberOfEvents(0),
	_numberOfUnorderedEvents(0),
	_kernelBuffer(nullptr),
	_temporalBuffer(nullptr),
	_lastTimestamp(0)
{
	struct perf_event_attr pe;
	pid_t pid;
	unsigned long flags;
	uint64_t alignedSize;
	void *perfMap;

	alignedSize = (userSize + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1);
	_metaSize   = PAGE_SIZE;
	_dataSize   = alignedSize;
	_totalSize  = _metaSize + _dataSize;
	_dataMask   = _dataSize - 1;
	_tempSize   = PAGE_SIZE;

	// Initialize common perf attribute argument
	memset(&pe, 0, sizeof(struct perf_event_attr));
	pe.type = PERF_TYPE_TRACEPOINT;
	pe.size = sizeof(struct perf_event_attr);
	pe.sample_period = 1; // record every n-events, setting to 1 records all
	pe.sample_type = PERF_SAMPLE_TIME | PERF_SAMPLE_RAW;
	pe.wakeup_events = 9999999;
	pe.disabled = 1;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 1, 0 )
	pe.use_clockid = 1;
	pe.clockid = CLOCK_MONOTONIC_RAW;
#endif

	// Tracing other than the current thread requires CAP_SYS_ADMIN
	// capability to set /proc/sys/kernel/perf_event_paranoid value less
	// than 1.
	pid = -1;
#ifdef PERF_FLAG_FD_CLOEXEC
	flags = PERF_FLAG_FD_CLOEXEC;
#else
	flags = 0;
#endif

	// Open each requested event file descriptor
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

	// Map kernel and user shared perf ring buffer
	perfMap = mmap(
		NULL, _totalSize,
		PROT_READ | PROT_WRITE, MAP_SHARED,
		_eventsFds[0], 0
	);
	if (perfMap == MAP_FAILED) {
		FatalErrorHandler::fail(
			"CTF: Kernel: When mapping perf pages: ",
			strerror(errno)
		);
	}

	// Change file descriptor properties
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

	// Allocate a temporal buffer used to copy events that have been split
	// betwen the end and beginning of the shared ring buffer
	_temporalBuffer = malloc(_tempSize);
	if (!_temporalBuffer) {
		FatalErrorHandler::fail(
			"CTF: Kernel: Cannot allocate memory for temporal buffer: ",
			strerror(errno)
		);
	}

	// Setup initial pointers
	_kernelBuffer = perfMap;
	_dataRing     = ((char *)perfMap) + PAGE_SIZE;
	_metaPage     = _kernelBuffer;

	struct perf_event_mmap_page *metaPage = (struct perf_event_mmap_page *) _metaPage;
	_localHead = metaPage->data_head;
	_localTail = metaPage->data_tail;
	_current = _localTail;

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 1, 0 )
	assert(metaPage->data_offset == PAGE_SIZE);
	assert(metaPage->data_size == _dataSize);
#endif

	initializeDebug();
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
	// bit long. Hence, it's safe to access it
	header = (struct perf_event_header *) (_dataRing + (current & _dataMask));

	// Adapted from perf tools/perf/util/mmap.c:perf_mmap__read()
	if (((current & _dataMask) + header->size) != ((current + header->size) & _dataMask)) {
		uint64_t offset = current;
		uint64_t len = header->size;
		uint64_t cpy;
		char *dst = (char *) _temporalBuffer;

		assert(len <= _tempSize);

		do {
			cpy = std::min(_dataSize - (offset & _dataMask), len);
			memcpy(dst, _dataRing + (offset & _dataMask), cpy);
			offset += cpy;
			dst += cpy;
			len -= cpy;
		} while (len);

		header = (struct perf_event_header *) _temporalBuffer;
	}

	return header;
}

bool CTFAPI::CTFKernelEventsProvider::read(void *buf, uint64_t size, uint64_t *read)
{
	struct perf_event_header *header;
	PerfRecordSample *perfRecordSample;
	PerfLostRecord *perfRecordLost;
	EventCommon *eventCommonFields;
	EventHeader *eventHeader;

	ctf_kernel_event_id_t eventId;
	ctf_kernel_event_size_t payloadSize;
	uint64_t eventSize = 0;
	uint64_t processedBytes = 0;
	uint64_t remainingBytes = size;
	char *buffer = (char *) buf;

	// Iterate over all events until either:
	//  1) we have processed all events
	//  2) There is not enough free space in the supplied buffer to process
	//     the current event
	while (_current != _localHead) {

		// Get the next record in the shared ring buffer
		header = (struct perf_event_header *) getNextEvent(_current);

		// Process record according to its type
		switch(header->type) {
		case PERF_RECORD_SAMPLE:
		{
			perfRecordSample = (PerfRecordSample *) header;
			eventCommonFields = (EventCommon *) perfRecordSample->data;
			eventId = eventCommonFields->common_type;
			payloadSize = (*_eventSizes)[eventId];
			assert(payloadSize <= perfRecordSample->size - sizeof(EventCommon));

			// Debug checks to verify integrity of the current event
			checkPerfRecordSampleSize(perfRecordSample);
			checkPerfEventSize(perfRecordSample, payloadSize);
			printKernelEvent(perfRecordSample, payloadSize);

			// Copy the event from it's original location (either
			// the kernel buffer or the temporal buffer) into the
			// user supplied buffer
			eventSize = _eventHeaderSize + payloadSize;
			if (eventSize <= remainingBytes) {
				// Copy event header
				eventHeader = (EventHeader *) buffer;
				eventHeader->id = eventId;
				eventHeader->timestamp = (perfRecordSample->time - _referenceTimestamp);

				// Check if this event is unordered
				if (perfRecordSample->time < _lastTimestamp) {
					_numberOfUnorderedEvents++;
				}
				_lastTimestamp = perfRecordSample->time;

				// Copy event payload
				memcpy(
					buffer + sizeof(EventHeader),
					perfRecordSample->data + sizeof(EventCommon),
					payloadSize
				);

				// Account written event
				buffer         += eventSize;
				remainingBytes -= eventSize;
				processedBytes += eventSize;
				_numberOfEvents++;
			} else {
				// No space left
				goto early_end;
			}

			break;
		}
		case PERF_RECORD_LOST:
		{
			// Some events have been lost, account them
			perfRecordLost = (PerfLostRecord *) header;
			_lost += perfRecordLost->lost;
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
			// The current record type is unknown to this
			// implementation. Demangle and report it if debug or
			// ignore it otherwise
			warnUnknownPerfRecord(header);
		}

		// If the event was copied sucessfully to the user buffer, move
		// on to the next event
		_current += header->size;
	}

early_end:

	if (processedBytes) {
		// Good ending. We have copied at least one event
		*read = processedBytes;
		return true;
	} else {
		// Bad ending. We have __not__ copied a single event. Return the
		// size of the event that we couldn't copy or 0 if there was
		// really no event
		*read = eventSize;
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

	_localHead = smpLoadAcquire(&metaPage->data_head);
}

void CTFAPI::CTFKernelEventsProvider::updateTail()
{
	struct perf_event_mmap_page *metaPage = (struct perf_event_mmap_page *) _metaPage;

	_localTail = _current;
	smpStoreRelease(&metaPage->data_tail, _localTail);
}
