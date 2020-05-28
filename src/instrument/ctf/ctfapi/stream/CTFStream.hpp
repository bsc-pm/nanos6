/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPUSTREAM_HPP
#define CPUSTREAM_HPP

#include <string>
#include <cstdint>
#include <string>
#include <map>

#include "../CTFTypes.hpp"
#include "../context/CTFContext.hpp"

namespace CTFAPI {
	class CTFStream {
	public:
		ctf_stream_id_t streamId;
		char *buffer;
		uint64_t bufferSize;
		uint64_t subBufferSize;
		uint64_t head;
		uint64_t tail;
		uint64_t mask;
		uint64_t subBufferMask;

		ctf_cpu_id_t cpuId;
		int fdOutput;
		off_t fileOffset;
		CTFContext *context;

		CTFStream() : streamId(0), bufferSize(0), context(nullptr) {}
		virtual ~CTFStream() {}

		void initialize(size_t size, ctf_cpu_id_t cpu);
		void shutdown(void);

		void flushAll();
		void flushFilledSubBuffers();
		bool checkIfNeedsFlush();
		bool checkFreeSpace(size_t size);

		virtual void lock() {}
		virtual void unlock() {}
		virtual void writeContext(__attribute__((unused)) void **buf) {}

		virtual size_t getContextSize(void) const
		{
			return 0;
		}

		void setContext(CTFContext *ctfcontext)
		{
			context = ctfcontext;
		}

	private:
		void *mrb;
		size_t mrbSize;

		void doWrite(int fd, const char *buf, size_t size);
	};
}

#endif //CPUSTREAM_HPP
