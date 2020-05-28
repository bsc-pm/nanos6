/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <string>
#include <iostream>
#include <cassert>
#include <errno.h>

#include <lowlevel/FatalErrorHandler.hpp>

#include "CTFAPI.hpp"
#include "CTFTrace.hpp"
#include "CTFTypes.hpp"
#include "CTFEvent.hpp"

extern CTFAPI::CTFEvent *__eventCTFFlush;


uint64_t CTFAPI::getTimestamp()
{
	uint64_t timestamp;
	struct timespec tp;
	const uint64_t ns = 1000000000ULL;

	if (clock_gettime(CLOCK_MONOTONIC, &tp)) {
		FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: clock_gettime syscall: ") + strerror(errno));
	}
	timestamp = tp.tv_sec * ns + tp.tv_nsec;

	return timestamp;
}

uint64_t CTFAPI::getRelativeTimestamp()
{
	uint64_t timestamp;
	CTFTrace &trace = CTFTrace::getInstance();

	timestamp = getTimestamp();
	timestamp -= trace.getAbsoluteStartTimestamp();

	return timestamp;
}

void CTFAPI::mk_event_header(char **buf, uint64_t timestamp, uint8_t id)
{
	struct event_header *pk;

	pk = (struct event_header *) *buf;
	*pk = (struct event_header) {
		.id = id,
		.timestamp = timestamp
	};

	*buf += sizeof(struct event_header);
}

static int mk_packet_header(char *buf, uint64_t *head, ctf_stream_id_t streamId)
{
	struct __attribute__((__packed__)) packet_header {
		uint32_t magic;
		ctf_stream_id_t stream_id;
	};

	const int pks = sizeof(struct packet_header);
	struct packet_header *pk;

	pk = (struct packet_header *) &buf[*head];
	*pk = (struct packet_header) {
		.magic = 0xc1fc1fc1,
		.stream_id = streamId
	};

	*head += pks;

	return 0;
}

static int mk_packet_context(char *buf, size_t *head, ctf_cpu_id_t cpu_id)
{
	struct __attribute__((__packed__)) packet_context {
		ctf_cpu_id_t cpu_id;
	};

	const int pks = sizeof(struct packet_context);
	struct packet_context *pk;

	pk = (struct packet_context *) &buf[*head];
	*pk = (struct packet_context) {
		.cpu_id = cpu_id,
	};

	*head += pks;

	return 0;
}

void CTFAPI::greetings(void)
{
	std::cout << "!!!!!!!!!!!!!!!!CTF API UP & Running!!!!!!!!!!!!!!!!" << std::endl;
}

void CTFAPI::addStreamHeader(CTFAPI::CTFStream *stream)
{
	// we don't need to mask the head because the buffer is at least 1 page
	// long and at this point it's empty
	mk_packet_header (stream->buffer, &stream->head, stream->streamId);
	mk_packet_context(stream->buffer, &stream->head, stream->cpuId);
}

void CTFAPI::writeFlushingTracepoint(CTFStream *stream,
				      uint64_t tsBefore, uint64_t tsAfter)
{
	uint64_t timestamp = getRelativeTimestamp();

	__tp_lock(stream, __eventCTFFlush, timestamp, tsBefore, tsAfter);
}

void CTFAPI::flushBuffer(CTFStream *stream,
			 uint64_t *tsBefore, uint64_t *tsAfter)
{
	*tsBefore = getRelativeTimestamp();
	stream->flushFilledSubBuffers();
	*tsAfter = getRelativeTimestamp();
}

void CTFAPI::flushCurrentVirtualCPUBufferIfNeeded()
{
	uint64_t tsBefore, tsAfter;
	CTFStream *stream = Instrument::getCPULocalData()->userStream;

	stream->lock();
	if (stream->checkIfNeedsFlush()) {
		flushBuffer(stream, &tsBefore, &tsAfter);
		writeFlushingTracepoint(stream, tsBefore, tsAfter);
	}
	stream->unlock();
}
