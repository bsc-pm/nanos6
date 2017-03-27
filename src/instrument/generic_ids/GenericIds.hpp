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
