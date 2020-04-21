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

#define xstr(s) str(s)
#define str(s) #s


int CTFAPI::mk_event_header(char **buf, uint8_t id)
{
	struct timespec tp;
	uint64_t timestamp;
	struct event_header *pk;
	const uint64_t ns = 1000000000ULL;
	CTFTrace &trace = CTFTrace::getInstance();

	pk = (struct event_header *) *buf;

	if (clock_gettime(CLOCK_MONOTONIC, &tp)) {
		FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: clock_gettime syscall: ") + strerror(errno));
	}
	timestamp = tp.tv_sec * ns + tp.tv_nsec;
	timestamp -= trace.getAbsoluteStartTimestamp();

	*pk = (struct event_header) {
		.id = id,
		.timestamp = timestamp
	};

	*buf += sizeof(struct event_header);

	return 0;
}

static int mk_packet_header(char *buf, uint64_t *head)
{
	struct __attribute__((__packed__)) packet_header {
		uint32_t magic;
		uint32_t stream_id;
	};

	const int pks = sizeof(struct packet_header);
	struct packet_header *pk;

	pk = (struct packet_header *) &buf[*head];
	*pk = (struct packet_header) {
		.magic = 0xc1fc1fc1,
		.stream_id = 0
	};

	*head += pks;

	return 0;
}

static int mk_packet_context(char *buf, size_t *head, uint16_t cpu_id)
{
	struct __attribute__((__packed__)) packet_context {
		uint16_t cpu_id;
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
	mk_packet_header (stream->buffer, &stream->head);
	mk_packet_context(stream->buffer, &stream->head, stream->cpuId);
}
