#ifndef INSTRUMENT_NULL_INSTRUMENTATION_CONTEXT_HPP
#define INSTRUMENT_NULL_INSTRUMENTATION_CONTEXT_HPP


#include <InstrumentComputePlaceId.hpp>
#include <InstrumentTaskId.hpp>
#include <InstrumentThreadId.hpp>

#include <cassert>


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
	
	
	//! \brief Creates a thread-local instrumentation context with the scope of the lifetime of the object itself
	class ThreadInstrumentationContext {
	private:
		static thread_local InstrumentationContext _context;
		
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


#endif // INSTRUMENT_NULL_INSTRUMENTATION_CONTEXT_HPP
