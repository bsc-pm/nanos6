/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPUSTREAM_HPP
#define CPUSTREAM_HPP

#include <string>
#include <cstdint>

#include "CircularBuffer.hpp"
#include "instrument/ctf/ctfapi/CTFTypes.hpp"
#include "instrument/ctf/ctfapi/context/CTFContext.hpp"

namespace CTFAPI {

	enum ctf_streams {
		CTFStreamKernelId    = 0,
		CTFStreamBoundedId   = 1,
		CTFStreamUnboundedId = 2
	};

	class CTFStream {

	private:

		int _node;
		ctf_stream_id_t _streamId;
		size_t _size;

		void addStreamHeader();
		void makePacketContext(CircularBuffer *circularBuffer, ctf_cpu_id_t cpu_id);
		void makePacketHeader(CircularBuffer *circularBuffer, ctf_stream_id_t streamId);

	protected:

		struct __attribute__((__packed__)) PacketHeader {
			uint32_t magic;
			ctf_stream_id_t stream_id;
		};

		// TODO possibly add index data here to speed up lookups
		struct __attribute__((__packed__)) PacketContext {
			ctf_cpu_id_t cpu_id;
		};

		ctf_cpu_id_t   _cpuId;
		std::string    _path;
		CircularBuffer _circularBuffer;

		CTFStream(size_t size, ctf_cpu_id_t cpu, int node, std::string path,
			  ctf_stream_id_t streamId);

	public:
		CTFStream(size_t size, ctf_cpu_id_t cpu, int node, std::string path)
			: CTFStream(size, cpu, node, path, CTFStreamBoundedId) {}

		virtual ~CTFStream() {}

		void initialize();

		virtual inline void shutdown()
		{
			_circularBuffer.shutdown();
		}

		virtual void lock() {}

		virtual void unlock() {}

		virtual void writeContext(void **) {}

		virtual size_t getContextSize() const
		{
			return 0;
		}

		inline ctf_stream_id_t getId() const
		{
			return _streamId;
		}

		inline ctf_cpu_id_t getCPUId() const
		{
			return _cpuId;
		}

		inline bool alloc(uint64_t size)
		{
			return _circularBuffer.alloc(size);
		}

		inline void *getBuffer()
		{
			return _circularBuffer.getBuffer();
		}

		inline void submit(uint64_t size)
		{
			_circularBuffer.submit(size);
		}

		inline bool checkIfNeedsFlush() {
			return _circularBuffer.checkIfNeedsFlush();
		}

		inline void flushFilledSubBuffers()
		{
			_circularBuffer.flushFilledSubBuffers();
		}

		inline void flushAll()
		{
			_circularBuffer.flushAll();
		}
	};
}

#endif //CPUSTREAM_HPP
