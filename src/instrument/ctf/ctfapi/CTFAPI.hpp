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

#include "CTFTypes.hpp"
#include "CTFEvent.hpp"
#include "stream/CTFStream.hpp"

namespace CTFAPI {

	struct __attribute__((__packed__)) event_header {
		ctf_event_id_t id;
		uint64_t timestamp;
	};

	void flushCurrentVirtualCPUBufferIfNeeded();

	// TODO isolate these into CTFAPI::core namespace
	uint64_t getTimestamp();
	uint64_t getRelativeTimestamp();
	void mk_event_header(char **buf, uint64_t timestamp, uint8_t id);
	void flushAll(CTFStream *stream, uint64_t *before, uint64_t *after);
	void flushSubBuffers(CTFStream *stream, uint64_t *before, uint64_t *after);
	void writeFlushingTracepoint(CTFStream *stream, uint64_t tsBefore, uint64_t tsAfter);

	template <typename T>
	static inline size_t sizeOfVariadic(T arg)
	{
		size_t value = sizeof(arg);
		return value;
	}

	template <>
	inline size_t sizeOfVariadic(const char *arg)
	{
		size_t i = 0;
		for (; arg[i]; ++i)
			;
		size_t value = i * sizeof(char);
		return value + 1; // adding 1 to count for the null character
	}

	template<typename First, typename... Rest>
	static inline size_t sizeOfVariadic(First arg, Rest... rest)
	{
		size_t total = sizeOfVariadic(arg) + sizeOfVariadic(rest...);
		return total;
	}

	template<typename... ARGS>
	static size_t eventSize(CTFStream *stream, CTFEvent *event, ARGS... args)
	{
		size_t size;

		size = sizeof(struct event_header) +
			stream->getContextSize() +
			event->getContextSize(stream->getId()) +
			sizeOfVariadic(args...);

		return size;
	}

	template<typename... ARGS>
	inline void tp_write_args(void **, ARGS...)
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
	inline void tp_write_args(void **buf, const char *arg)
	{
		char **pbuf = reinterpret_cast<char**>(buf);
		char *parg = (char *) arg;

		while (*parg != '\0') {
			**pbuf = *parg;
			parg++;
			(*pbuf)++;
		}

		**pbuf = '\0';
		(*pbuf)++;
	}

	template<typename T, typename... ARGS>
	static inline void tp_write_args(void **buf, T arg, ARGS... args)
	{
		tp_write_args(buf, arg);
		tp_write_args(buf, args...);
	}

	template<typename... ARGS>
	static void __tp_lock_write(CTFStream *stream, CTFEvent *event, uint64_t timestamp, size_t size, ARGS... args)
	{
		const uint8_t tracepointId = event->getEventId();
		void *buf = stream->getBuffer();

		mk_event_header((char **) &buf, timestamp, tracepointId);
		stream->writeContext(&buf);
		event->writeContext(&buf, stream->getId());
		tp_write_args(&buf, args...);
		stream->submit(size);
	}

	template<typename... ARGS>
	static void __tp_lock(CTFStream *stream, CTFEvent *event, uint64_t timestamp, ARGS... args)
	{
		size_t size;
		bool needsFlush;
		uint64_t tsBefore = 0;
		uint64_t tsAfter  = 0;

		// calculate the total size of this tracepoint
		size = eventSize(stream, event, args...);

		// check if there is enough free space to write this tracepoint
		needsFlush = !stream->alloc(size);

		// if not, flush buffers first and record when the flushing
		// started and finished
		if (needsFlush)
			flushAll(stream, &tsBefore, &tsAfter);

		// write the tracepoint (notice that we took this tracepoint
		// timestamp before any flushing was possibly made)
		__tp_lock_write(stream, event, timestamp, size, args...);

		// if we flused, write the start and end flushing tracepoints as
		// they originally ocurred. Writing the flushing tracepoints
		// will call this function recursively. As long as the buffer
		// can hold two flush tracepoints, this function will be called
		// at most two extra times
		if (needsFlush)
			writeFlushingTracepoint(stream, tsBefore, tsAfter);
	}

	// TODO add nanos6 developer instructions for adding tracepoints
	template<typename... ARGS>
	static void tracepoint_async(CTFEvent *event, uint64_t timestamp, ARGS... args)
	{
		// When issuing async tracepoints, we cannot rely on locks
		// because the timestamp should be aquired within a locked
		// region. Otherwise we could be writing unordered events and
		// corrupting the trace. However, here the user is providing us
		// with a timestamp obtained in the past.  Hence, the
		// tracepoint_async user must guarantee that events are written
		// sequentially

		CTFStream *stream = Instrument::getCPULocalData()->userStream;
		__tp_lock(stream, event, timestamp, args...);
	}

	// TODO add nanos6 developer instructions for adding tracepoints
	template<typename... ARGS>
	static void tracepoint(CTFEvent *event, ARGS... args)
	{
		// Obtaining the per-cpu local object might trigger a
		// tracepoint. This will happen if the current thread is an
		// external thread that has not been initialized yet. In that
		// case a nanos6:external_thread_create event will be emited
		// before the current one
		CTFStream *stream = Instrument::getCPULocalData()->userStream;

		// Locking only implemented for external threads
		stream->lock();

		uint64_t timestamp = getRelativeTimestamp();
		__tp_lock(stream, event, timestamp, args...);

		stream->unlock();
	}
}

#endif // CTFAPI_HPP
