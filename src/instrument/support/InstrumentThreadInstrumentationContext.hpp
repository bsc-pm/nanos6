/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_THREAD_INSTRUMENTATION_CONTEXT_HPP
#define INSTRUMENT_SUPPORT_THREAD_INSTRUMENTATION_CONTEXT_HPP


#include <InstrumentInstrumentationContext.hpp>

#include <string>


namespace Instrument {
	//! \brief Creates a thread-local instrumentation context with the scope of the lifetime of the object itself
	class ThreadInstrumentationContext {
	private:
		InstrumentationContext _oldContext;
		
	public:
		ThreadInstrumentationContext(task_id_t const &taskId, compute_place_id_t const &computePlaceId, thread_id_t const &threadId);
		ThreadInstrumentationContext(task_id_t const &taskId);
		ThreadInstrumentationContext(compute_place_id_t const &computePlaceId);
		ThreadInstrumentationContext(thread_id_t const &threadId);
		ThreadInstrumentationContext(std::string const *externalThreadName);
		
		~ThreadInstrumentationContext();
		
		InstrumentationContext const &get() const;
		
		static InstrumentationContext const &getCurrent();
		static void updateComputePlace(compute_place_id_t const &computePlaceId);
	};
}


#endif // INSTRUMENT_SUPPORT_THREAD_INSTRUMENTATION_CONTEXT_HPP
