/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_INSTRUMENTATION_CONTEXT_HPP
#define INSTRUMENT_SUPPORT_INSTRUMENTATION_CONTEXT_HPP


#include <InstrumentComputePlaceId.hpp>
#include <InstrumentTaskId.hpp>
#include <InstrumentThreadId.hpp>

#include <cassert>
#include <string>


namespace Instrument {
	//! \brief Data needed by the instrumentation API
	struct InstrumentationContext {
		task_id_t _taskId;
		compute_place_id_t _computePlaceId;
		thread_id_t _threadId;
		std::string const *_externalThreadName;
		
		InstrumentationContext()
			: _externalThreadName(nullptr)
		{
		}
		
		InstrumentationContext(task_id_t const &taskId, compute_place_id_t const &computePlaceId, thread_id_t const &threadId)
			: _taskId(taskId), _computePlaceId(computePlaceId), _threadId(threadId), _externalThreadName(nullptr)
		{
		}
		
		InstrumentationContext(InstrumentationContext const &other)
			: _taskId(other._taskId), _computePlaceId(other._computePlaceId), _threadId(other._threadId),
			_externalThreadName(other._externalThreadName)
		{
		}
		
		InstrumentationContext(std::string const *externalThreadName)
			: _externalThreadName(externalThreadName)
		{
		}
		
		bool empty() const
		{
			return (_taskId == task_id_t()) && (_computePlaceId == compute_place_id_t()) && (_threadId == thread_id_t()) && (_externalThreadName == nullptr);
		}
		
		bool operator==(InstrumentationContext const &other) const
		{
			return (_taskId == other._taskId) && (_computePlaceId == other._computePlaceId) && (_threadId == other._threadId) && (_externalThreadName == other._externalThreadName);
		}
		
		bool operator!=(InstrumentationContext const &other) const
		{
			return !(*this == other);
		}
	};
	
	
	//! \brief A non-thread-local instrumentation 
	typedef InstrumentationContext LocalInstrumentationContext;
}


#endif // INSTRUMENT_SUPPORT_INSTRUMENTATION_CONTEXT_HPP
