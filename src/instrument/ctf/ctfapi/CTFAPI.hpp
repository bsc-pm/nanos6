/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFAPI_HPP
#define CTFAPI_HPP

#include <cstdint>
#include <inttypes.h>
#include <time.h>
#include <string.h>

#include "InstrumentCPULocalData.hpp"
#include "instrument/support/InstrumentCPULocalDataSupport.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#include "CTFEvent.hpp"
#include "CTFStream.hpp"

typedef uint16_t ctf_task_type_id_t;
typedef uint32_t ctf_task_id_t;
typedef uint16_t ctf_cpu_id_t; //TODO use this typedef for all cpu ids variables

namespace CTFAPI {

	#define ARG_STRING_SIZE 64

	struct __attribute__((__packed__)) event_header {
		uint8_t  id;
		uint64_t timestamp;
	};

	void greetings(void);
	void addStreamHeader(CTFStream *stream);
	int mk_event_header(char **buf, uint8_t id);


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

	template <>
	struct sizeOfVariadic<char *>
	{
	    static constexpr size_t value = ARG_STRING_SIZE;
	};

	template<typename... ARGS>
	inline void tp_write_args(void **buf, ARGS... args)
	{
	}

	template<typename T>
	static inline void tp_write_args(void **buf, T arg)
	{
		T **p = reinterpret_cast<T**>(buf);
		**p = arg;
		(*p)++;
	}

	template<>
	inline void tp_write_args(void **buf, char *arg)
	{
		const int MAX = ARG_STRING_SIZE;
		int cnt = 0;
		int padding;
		char **pbuf = reinterpret_cast<char**>(buf);
		char *parg = arg;

		// copy string until the null character is found or we
		// reach the MAX limit
		while ((*parg != '\0') && (cnt < (MAX-1))) {
			**pbuf = *parg;
			parg++;
			(*pbuf)++;
			cnt++;
		}
		// add padding until we reach ARG_STRING_SIZE
		padding = MAX - cnt; // includes final null character
		memset(*pbuf, '\0', padding);
		*pbuf += padding;
	}

	template<typename T, typename... ARGS>
	static inline void tp_write_args(void **buf, T arg, ARGS... args)
	{
		tp_write_args(buf, arg);
		tp_write_args(buf, args...);
	}

	// TODO update instructions
	//
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
	static void tracepoint(CTFEvent *event, ARGS... args)
	{
		CTFStream *stream = Instrument::getCPULocalData()->userStream;
		const size_t size = sizeof(struct event_header) + sizeOfVariadic<ARGS...>::value + event->getContextSize();
		const uint8_t tracepointId = event->getEventId();
		void *buf;

		stream->lock();

		// TODO checkFreeSpace should not perform flushing, move
		// it here.
		// TODO add flushing tracepoints if possible
		if (!stream->checkFreeSpace(size)) {
			stream->unlock();
			return;
		}

		buf = stream->buffer + (stream->head & stream->mask);

		mk_event_header((char **) &buf, tracepointId);
		event->writeContext(&buf);
		tp_write_args(&buf, args...);

		stream->head += size;

		stream->unlock();
	}
}

#endif // CTFAPI_HPP
