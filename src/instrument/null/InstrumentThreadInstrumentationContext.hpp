/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_THREAD_INSTRUMENTATION_CONTEXT_HPP
#define INSTRUMENT_NULL_THREAD_INSTRUMENTATION_CONTEXT_HPP


#include <InstrumentInstrumentationContext.hpp>


namespace Instrument {
	//! \brief Creates a thread-local instrumentation context with the scope of the lifetime of the object itself
	class ThreadInstrumentationContext {
	public:
		ThreadInstrumentationContext(__attribute__((unused)) task_id_t const &taskId, __attribute__((unused)) compute_place_id_t const &computePlaceId, __attribute__((unused)) thread_id_t const &threadId)
		{
		}
		
		ThreadInstrumentationContext(__attribute__((unused)) task_id_t const &taskId)
		{
		}
		
		ThreadInstrumentationContext(__attribute__((unused)) compute_place_id_t const &computePlaceId)
		{
		}
		
		ThreadInstrumentationContext(__attribute__((unused)) thread_id_t const &threadId)
		{
		}
		
		~ThreadInstrumentationContext()
		{
		}
		
		InstrumentationContext get() const
		{
			return InstrumentationContext();
		}
		
		static InstrumentationContext getCurrent()
		{
			return InstrumentationContext();
		}
		
		static void updateComputePlace(__attribute__((unused)) compute_place_id_t const &computePlaceId)
		{
		}
	};
	
}


#endif // INSTRUMENT_NULL_THREAD_INSTRUMENTATION_CONTEXT_HPP
