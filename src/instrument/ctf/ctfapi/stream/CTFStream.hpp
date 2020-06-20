/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPUSTREAM_HPP
#define CPUSTREAM_HPP

#include <string>
#include <cstdint>

#include "CircularBuffer.hpp"
#include "../CTFTypes.hpp"
#include "../context/CTFContext.hpp"

namespace CTFAPI {

	enum ctf_streams {
		CTFStreamBoundedId   = 1,
		CTFStreamUnboundedId = 2
	};

	class CTFStream {

	private:
		ctf_cpu_id_t    _cpuId;
		ctf_stream_id_t _streamId;
		CircularBuffer _circularBuffer;

		void addStreamHeader();

	protected:
		CTFStream(size_t size, ctf_cpu_id_t cpu, std::string path,
			  ctf_stream_id_t streamId);

	public:
		CTFStream(size_t size, ctf_cpu_id_t cpu, std::string path)
			: CTFStream(size, cpu, path, CTFStreamBoundedId) {}

		virtual ~CTFStream() {}

		inline void shutdown()
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
