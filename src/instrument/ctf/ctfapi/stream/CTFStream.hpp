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
	class CTFStream {

	private:
		ctf_cpu_id_t    _cpuId;
		ctf_stream_id_t _streamId;
		CircularBuffer circularBuffer;

		void addStreamHeader();

	protected:
		CTFStream(size_t size, ctf_cpu_id_t cpu, std::string path,
			  ctf_stream_id_t streamId);

	public:
		CTFContext *context;

		CTFStream(size_t size, ctf_cpu_id_t cpu, std::string path)
			: CTFStream(size, cpu, path, 0) {}

		virtual ~CTFStream() {}

		inline void shutdown()
		{
			circularBuffer.shutdown();
		}

		virtual void lock() {}
		virtual void unlock() {}
		virtual void writeContext(__attribute__((unused)) void **buf) {}

		virtual size_t getContextSize() const
		{
			return 0;
		}

		inline void setContext(CTFContext *ctfcontext)
		{
			context = ctfcontext;
		}

		inline bool alloc(uint64_t size)
		{
			return circularBuffer.alloc(size);
		}

		inline void *getBuffer()
		{
			return circularBuffer.getBuffer();
		}

		inline void submit(uint64_t size)
		{
			circularBuffer.submit(size);
		}

		inline bool checkIfNeedsFlush() {
			return circularBuffer.checkIfNeedsFlush();
		}

		inline void flushFilledSubBuffers()
		{
			circularBuffer.flushFilledSubBuffers();
		}

		inline void flushAll()
		{
			circularBuffer.flushAll();
		}
	};
}

#endif //CPUSTREAM_HPP
