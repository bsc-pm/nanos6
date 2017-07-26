/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_THREAD_ID_HPP
#define INSTRUMENT_NULL_THREAD_ID_HPP


namespace Instrument {
	//! This is the default thread identifier for the instrumentation.
	//! It should be redefined in an identically named file within each instrumentation implementation.
	struct thread_id_t {
		bool operator==(__attribute__((unused)) thread_id_t const &other) const
		{
			return true;
		}
	};
}


#endif // INSTRUMENT_NULL_THREAD_ID_HPP
