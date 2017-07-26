/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_INSTRUMENTATION_CONTEXT_HPP
#define INSTRUMENT_NULL_INSTRUMENTATION_CONTEXT_HPP


#include <InstrumentComputePlaceId.hpp>
#include <InstrumentTaskId.hpp>
#include <InstrumentThreadId.hpp>


namespace Instrument {
	//! \brief Data needed by the instrumentation API
	struct InstrumentationContext {
		InstrumentationContext()
		{
		}
		
		InstrumentationContext(__attribute__((unused)) task_id_t const &taskId, __attribute__((unused)) compute_place_id_t const &computePlaceId, __attribute__((unused)) thread_id_t const &threadId)
		{
		}
		
		InstrumentationContext(__attribute__((unused)) InstrumentationContext const &other)
		{
		}
		
		bool empty() const
		{
			return true;
		}
	};
	
	
	//! \brief A non-thread-local instrumentation 
	typedef InstrumentationContext LocalInstrumentationContext;
}


#endif // INSTRUMENT_NULL_INSTRUMENTATION_CONTEXT_HPP
