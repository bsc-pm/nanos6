/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_GENERIC_IDS_HPP
#define INSTRUMENT_GENERIC_IDS_HPP


#include "InstrumentExternalThreadId.hpp"
#include "InstrumentThreadId.hpp"

#include <atomic>


namespace Instrument {
	namespace GenericIds {
		extern std::atomic<thread_id_t::inner_type_t> _nextThreadId;
		extern std::atomic<external_thread_id_t::inner_type_t> _nextExternalThreadId;
		
		
		inline thread_id_t getNewThreadId()
		{
			thread_id_t::inner_type_t threadId = _nextThreadId++;
			return thread_id_t(threadId);
		}
		inline thread_id_t::inner_type_t getTotalThreads()
		{
			return _nextThreadId.load();
		}
		
		inline external_thread_id_t getNewExternalThreadId()
		{
			external_thread_id_t::inner_type_t threadId = _nextExternalThreadId++;
			return external_thread_id_t(threadId);
		}
		
		inline external_thread_id_t getCommonPoolNewExternalThreadId()
		{
			external_thread_id_t::inner_type_t threadId = _nextThreadId++;
			return external_thread_id_t(threadId);
		}
		
		inline thread_id_t::inner_type_t getTotalExternalThreads()
		{
			return _nextExternalThreadId.load();
		}
	}
}


#endif // INSTRUMENT_GENERIC_IDS_HPP
