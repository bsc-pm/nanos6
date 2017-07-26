/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GENERIC_IDS_HPP
#define INSTRUMENT_GENERIC_IDS_HPP


#include "InstrumentThreadId.hpp"

#include <atomic>


namespace Instrument {
	namespace GenericIds {
		extern std::atomic<thread_id_t::inner_type_t> _nextThreadId;
		
		
		inline thread_id_t getNewThreadId()
		{
			thread_id_t::inner_type_t threadId = _nextThreadId++;
			return thread_id_t(threadId);
		}
		
		inline thread_id_t::inner_type_t getTotalThreads()
		{
			return _nextThreadId.load();
		}
	}
}


#endif // INSTRUMENT_GENERIC_IDS_HPP
