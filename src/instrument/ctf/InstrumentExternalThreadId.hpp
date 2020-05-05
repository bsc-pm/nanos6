/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_EXTERNAL_THREAD_ID_HPP
#define INSTRUMENT_CTF_EXTERNAL_THREAD_ID_HPP

#include "ctfapi/CTFTypes.hpp"

namespace Instrument {
	//! This is the default non-worker thread identifier for the instrumentation.
	//! It should be redefined in an identically named file within each instrumentation implementation.
	struct external_thread_id_t {
		ctf_thread_id_t tid;

		external_thread_id_t() : tid(0) {}
		external_thread_id_t(ctf_thread_id_t threadId) : tid(threadId) {}

		bool operator==(external_thread_id_t const &other) const
		{
			return tid == other.tid;
		}
	};
}


#endif // INSTRUMENT_CTF_EXTERNAL_THREAD_ID_HPP
