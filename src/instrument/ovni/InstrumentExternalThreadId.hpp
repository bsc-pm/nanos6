/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_EXTERNAL_THREAD_ID_HPP
#define INSTRUMENT_OVNI_EXTERNAL_THREAD_ID_HPP

#include <cstdint>

namespace Instrument {
	//! This is the default non-worker thread identifier for the instrumentation.
	//! It should be redefined in an identically named file within each instrumentation implementation.
	struct external_thread_id_t {
		uint32_t tid;

		external_thread_id_t() : tid(0) {}
		external_thread_id_t(uint32_t threadId) : tid(threadId) {}

		bool operator==(external_thread_id_t const &other) const
		{
			return tid == other.tid;
		}
	};
}


#endif // INSTRUMENT_OVNI_EXTERNAL_THREAD_ID_HPP
