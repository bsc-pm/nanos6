#ifndef INSTRUMENT_SUPPORT_THREAD_INSTRUMENTATION_CONTEXT_HPP
#define INSTRUMENT_SUPPORT_THREAD_INSTRUMENTATION_CONTEXT_HPP


#include <InstrumentInstrumentationContext.hpp>


namespace Instrument {
	//! \brief Creates a thread-local instrumentation context with the scope of the lifetime of the object itself
	class ThreadInstrumentationContext {
	private:
		InstrumentationContext _oldContext;
		
	public:
		inline ThreadInstrumentationContext(task_id_t const &taskId, compute_place_id_t const &computePlaceId, thread_id_t const &threadId);
		inline ThreadInstrumentationContext(task_id_t const &taskId);
		inline ThreadInstrumentationContext(compute_place_id_t const &computePlaceId);
		inline ThreadInstrumentationContext(thread_id_t const &threadId);
		
		inline ~ThreadInstrumentationContext();
		
		inline InstrumentationContext const &get() const;
		
		inline static InstrumentationContext const &getCurrent();
		inline static void updateComputePlace(compute_place_id_t const &computePlaceId);
	};
}


#endif // INSTRUMENT_SUPPORT_THREAD_INSTRUMENTATION_CONTEXT_HPP
