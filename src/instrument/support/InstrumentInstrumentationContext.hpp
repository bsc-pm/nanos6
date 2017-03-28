#ifndef INSTRUMENT_SUPPORT_INSTRUMENTATION_CONTEXT_HPP
#define INSTRUMENT_SUPPORT_INSTRUMENTATION_CONTEXT_HPP


#include <InstrumentHardwarePlaceId.hpp>
#include <InstrumentTaskId.hpp>
#include <InstrumentThreadId.hpp>

#include <cassert>


namespace Instrument {
	//! \brief Data needed by the instrumentation API
	struct InstrumentationContext {
		task_id_t _taskId;
		hardware_place_id_t _hardwarePlaceId;
		thread_id_t _threadId;
		
		InstrumentationContext()
		{
		}
		
		InstrumentationContext(task_id_t const &taskId, hardware_place_id_t const &hardwarePlaceId, thread_id_t const &threadId)
			: _taskId(taskId), _hardwarePlaceId(hardwarePlaceId), _threadId(threadId)
		{
		}
		
		InstrumentationContext(InstrumentationContext const &other)
			: _taskId(other._taskId), _hardwarePlaceId(other._hardwarePlaceId), _threadId(other._threadId)
		{
		}
		
		bool empty() const
		{
			return (_taskId == task_id_t()) && (_hardwarePlaceId == hardware_place_id_t()) && (_threadId == thread_id_t());
		}
	};
	
	
	//! \brief A non-thread-local instrumentation 
	typedef InstrumentationContext LocalInstrumentationContext;
	
	
	//! \brief Creates a thread-local instrumentation context with the scope of the lifetime of the object itself
	class ThreadInstrumentationContext {
	private:
		static thread_local InstrumentationContext _context;
		
		InstrumentationContext _oldContext;
		
	public:
		ThreadInstrumentationContext(task_id_t const &taskId, hardware_place_id_t const &hardwarePlaceId, thread_id_t const &threadId)
		{
			_oldContext = _context;
			_context = InstrumentationContext(taskId, hardwarePlaceId, threadId);
		}
		
		ThreadInstrumentationContext(task_id_t const &taskId)
		{
			_oldContext = _context;
			_context = InstrumentationContext(taskId, _oldContext._hardwarePlaceId, _oldContext._threadId);
		}
		
		ThreadInstrumentationContext(hardware_place_id_t const &hardwarePlaceId)
		{
			_oldContext = _context;
			_context = InstrumentationContext(_oldContext._taskId, hardwarePlaceId, _oldContext._threadId);
		}
		
		ThreadInstrumentationContext(thread_id_t const &threadId)
		{
			_oldContext = _context;
			_context = InstrumentationContext(_oldContext._taskId, _oldContext._hardwarePlaceId, threadId);
		}
		
		~ThreadInstrumentationContext()
		{
			_context = _oldContext;
		}
		
		InstrumentationContext const &get() const
		{
			return _context;
		}
		
		static InstrumentationContext const &getCurrent()
		{
			return _context;
		}
		
		static void updateHardwarePlace(hardware_place_id_t const &hardwarePlaceId)
		{
			_context._hardwarePlaceId = hardwarePlaceId;
		}
	};
	
}


#endif // INSTRUMENT_SUPPORT_INSTRUMENTATION_CONTEXT_HPP
