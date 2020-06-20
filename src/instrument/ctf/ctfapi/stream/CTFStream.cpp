/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "CTFStream.hpp"

CTFAPI::CTFStream::CTFStream(size_t size, ctf_cpu_id_t cpu, std::string path,
			     ctf_stream_id_t streamId)
{
	_cpuId = cpu;
	_streamId = streamId;
	std::string streamFilePath = path + "/channel_" + std::to_string(_cpuId);
	_circularBuffer.initialize(size, streamFilePath.c_str());
	addStreamHeader();
}

static void mk_packet_header(CircularBuffer *circularBuffer, ctf_stream_id_t streamId)
{
	struct __attribute__((__packed__)) packet_header {
		uint32_t magic;
		ctf_stream_id_t stream_id;
	};

	const int pks = sizeof(struct packet_header);
	struct packet_header *pk;
	void *buf = circularBuffer->getBuffer();

	pk = (struct packet_header *) buf;
	*pk = (struct packet_header) {
		.magic = 0xc1fc1fc1,
		.stream_id = streamId
	};

	circularBuffer->submit(pks);
}

static void mk_packet_context(CircularBuffer *circularBuffer, ctf_cpu_id_t cpu_id)
{
	// TODO possibly add index data here to speed up lookups
	struct __attribute__((__packed__)) packet_context {
		ctf_cpu_id_t cpu_id;
	};

	const int pks = sizeof(struct packet_context);
	struct packet_context *pk;
	void *buf = circularBuffer->getBuffer();

	pk = (struct packet_context *) buf;
	*pk = (struct packet_context) {
		.cpu_id = cpu_id,
	};

	circularBuffer->submit(pks);
}

void CTFAPI::CTFStream::addStreamHeader()
{
	mk_packet_header (&_circularBuffer, _streamId);
	mk_packet_context(&_circularBuffer, _cpuId);
}
