/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFAPI_HPP
#define CTFAPI_HPP

#include <stdint.h>
#include <inttypes.h>
#include <time.h>

#include <InstrumentCPULocalData.hpp>
#include <instrument/support/InstrumentCPULocalDataSupport.hpp>

namespace CTFAPI {

	#define TP_NANOS6_TASK_EXECUTE 10000U
	#define TP_NANOS6_TASK_END     10001U
	#define TP_NANOS6_TASK_BLOCK   10002U

	struct __attribute__((__packed__)) event_header {
		uint32_t id;
		uint64_t timestamp;
	};

	void greetings(void);

	void addStreamHeader(Instrument::CTFStream &stream);
	void writeUserMetadata(std::string directory);
	void writeKernelMetadata(std::string directory);



	static int mk_event_header(char **buf, uint32_t id)
	{
		struct timespec tp;
		uint64_t timestamp;
		struct event_header *pk;
		const uint64_t ns = 1000000000ULL;

		pk = (struct event_header *) *buf;

		 //TODO use relative instead of absolute timestamp
		if (clock_gettime(CLOCK_MONOTONIC, &tp)) {
			FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: clock_gettime syscall: ") + strerror(errno));
		}
		timestamp = tp.tv_sec * ns + tp.tv_nsec;

		*pk = (struct event_header) {
			.id = id,
			.timestamp = timestamp
		};

		*buf += sizeof(struct event_header);

		return 0;
	}

	template<typename First, typename... Rest>
	struct sizeOfVariadic
	{
	    static constexpr size_t value = sizeOfVariadic<First>::value + sizeOfVariadic<Rest...>::value;
	};

	template <typename T>
	struct sizeOfVariadic<T>
	{
	    static constexpr size_t value = sizeof(T);
	};

	template<typename T>
	static inline void tp_write_args(void **buf, T arg)
	{
		T **p = reinterpret_cast<T**>(buf);
		**p = arg;
		(*p)++;
	}

	template<typename T, typename... ARGS>
	static inline void tp_write_args(void **buf, T arg, ARGS... args)
	{
		tp_write_args(buf, arg);
		tp_write_args(buf, args...);
	}

	// To add a new user-space tracepoint into Nanos6:
	//   1) add a new TP_NANOS6_* macro with your tracepoint id.
	//   2) add the corresponding metadata entry under CTFAPI.cpp with the
        //      previous ID. Define arguments as needed.
	//   3) call this function with the tracepoint ID as first argument and
        //      the correspnding arguments as defined in the metadata file,
        //      in the same order.
	// When calling this function, allways cast each variadic argument to
	// the type specified in the metadata file. Otherwhise an incorrect
	// number of bytes might be written.

	template<typename... ARGS>
	static void tracepoint(const uint32_t tracepointId, ARGS... args)
	{
		Instrument::CTFStream &stream = Instrument::getCPULocalData().userStream;
		const size_t size = sizeof(struct event_header) + sizeOfVariadic<ARGS...>::value;
		void *buf;

		if (!stream.checkFreeSpace(size))
			return;

		buf = stream.buffer + (stream.head & stream.mask);

		mk_event_header((char **) &buf, tracepointId);
		tp_write_args(&buf, args...);

		stream.head += size;
	}
}

#endif // CTFAPI_HPP
