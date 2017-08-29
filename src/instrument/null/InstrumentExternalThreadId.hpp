/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_EXTERNAL_THREAD_ID_HPP
#define INSTRUMENT_NULL_EXTERNAL_THREAD_ID_HPP


namespace Instrument {
	//! This is the default non-worker thread identifier for the instrumentation.
	//! It should be redefined in an identically named file within each instrumentation implementation.
	struct external_thread_id_t {
		bool operator==(__attribute__((unused)) external_thread_id_t const &other) const
		{
			return true;
		}
	};
}


#endif // INSTRUMENT_NULL_EXTERNAL_THREAD_ID_HPP
