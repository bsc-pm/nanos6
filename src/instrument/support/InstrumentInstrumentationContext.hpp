#ifndef INSTRUMENT_SUPPORT_INSTRUMENTATION_CONTEXT_HPP
#define INSTRUMENT_SUPPORT_INSTRUMENTATION_CONTEXT_HPP


#include <InstrumentComputePlaceId.hpp>
#include <InstrumentTaskId.hpp>
#include <InstrumentThreadId.hpp>

#include <cassert>


namespace Instrument {
	//! \brief Data needed by the instrumentation API
	struct InstrumentationContext {
		task_id_t _taskId;
		compute_place_id_t _computePlaceId;
		thread_id_t _threadId;
		
		InstrumentationContext()
		{
		}
		
		InstrumentationContext(task_id_t const &taskId, compute_place_id_t const &computePlaceId, thread_id_t const &threadId)
			: _taskId(taskId), _computePlaceId(computePlaceId), _threadId(threadId)
		{
		}
		
		InstrumentationContext(InstrumentationContext const &other)
			: _taskId(other._taskId), _computePlaceId(other._computePlaceId), _threadId(other._threadId)
		{
		}
		
		bool empty() const
		{
			return (_taskId == task_id_t()) && (_computePlaceId == compute_place_id_t()) && (_threadId == thread_id_t());
		}
	};
	
	
	//! \brief A non-thread-local instrumentation 
	typedef InstrumentationContext LocalInstrumentationContext;
}


#endif // INSTRUMENT_SUPPORT_INSTRUMENTATION_CONTEXT_HPP
