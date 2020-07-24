/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "CTFStream.hpp"

CTFAPI::CTFStream::CTFStream(size_t size, ctf_cpu_id_t cpu, std::string path,
			     ctf_stream_id_t streamId)
	: _cpuId(cpu),
	_streamId(streamId),
	_size(size)
{
	_path = path + "/channel_" + std::to_string(_cpuId);
}

void CTFAPI::CTFStream::initialize()
{
	_circularBuffer.initialize(_size, _path.c_str());
	addStreamHeader();
}

void CTFAPI::CTFStream::makePacketHeader(CircularBuffer *circularBuffer, ctf_stream_id_t streamId)
{
	const int pks = sizeof(struct PacketHeader);
	struct PacketHeader *pk;
	void *buf = circularBuffer->getBuffer();

	pk = (struct PacketHeader *) buf;
	*pk = (struct PacketHeader) {
		.magic = 0xc1fc1fc1,
		.stream_id = streamId
	};

	circularBuffer->submit(pks);
}

void CTFAPI::CTFStream::makePacketContext(CircularBuffer *circularBuffer, ctf_cpu_id_t cpu_id)
{
	const int pks = sizeof(struct PacketContext);
	struct PacketContext *pk;
	void *buf = circularBuffer->getBuffer();

	pk = (struct PacketContext *) buf;
	*pk = (struct PacketContext) {
		.cpu_id = cpu_id,
	};

	circularBuffer->submit(pks);
}

void CTFAPI::CTFStream::addStreamHeader()
{
	makePacketHeader(&_circularBuffer, _streamId);
	makePacketContext(&_circularBuffer, _cpuId);
}
