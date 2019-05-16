/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/


#include "InstrumentThreadLocalDataSupport.hpp"

#include <InstrumentThreadInstrumentationContext.hpp>


Instrument::ThreadInstrumentationContext::ThreadInstrumentationContext(task_id_t const &taskId, compute_place_id_t const &computePlaceId, thread_id_t const &threadId)
{
	Instrument::ThreadLocalData &threadLocal = Instrument::getThreadLocalData();
	
	_oldContext = threadLocal._context;
	threadLocal._context = InstrumentationContext(taskId, computePlaceId, threadId);
}

Instrument::ThreadInstrumentationContext::ThreadInstrumentationContext(task_id_t const &taskId)
{
	Instrument::ThreadLocalData &threadLocal = Instrument::getThreadLocalData();
	
	_oldContext = threadLocal._context;
	threadLocal._context = InstrumentationContext(taskId, _oldContext._computePlaceId, _oldContext._threadId);
}

Instrument::ThreadInstrumentationContext::ThreadInstrumentationContext(compute_place_id_t const &computePlaceId)
{
	Instrument::ThreadLocalData &threadLocal = Instrument::getThreadLocalData();
	
	_oldContext = threadLocal._context;
	threadLocal._context = InstrumentationContext(_oldContext._taskId, computePlaceId, _oldContext._threadId);
}

Instrument::ThreadInstrumentationContext::ThreadInstrumentationContext(thread_id_t const &threadId)
{
	Instrument::ThreadLocalData &threadLocal = Instrument::getThreadLocalData();
	
	_oldContext = threadLocal._context;
	threadLocal._context = InstrumentationContext(_oldContext._taskId, _oldContext._computePlaceId, threadId);
}

Instrument::ThreadInstrumentationContext::ThreadInstrumentationContext(std::string const *externalThreadName)
{
	Instrument::ThreadLocalData &threadLocal = Instrument::getThreadLocalData();
	
	_oldContext = threadLocal._context;
	threadLocal._context = InstrumentationContext(externalThreadName);
}

Instrument::ThreadInstrumentationContext::~ThreadInstrumentationContext()
{
	Instrument::ThreadLocalData &threadLocal = Instrument::getThreadLocalData();
	
	threadLocal._context = _oldContext;
}

Instrument::InstrumentationContext const &Instrument::ThreadInstrumentationContext::get() const
{
	Instrument::ThreadLocalData &threadLocal = Instrument::getThreadLocalData();
	if (&threadLocal != &Instrument::getSentinelNonWorkerThreadLocalData()) {
		return threadLocal._context;
	} else {
		Instrument::ExternalThreadLocalData &externalThreadLocal = Instrument::getExternalThreadLocalData();
		return externalThreadLocal._context;
	}
}

Instrument::InstrumentationContext const &Instrument::ThreadInstrumentationContext::getCurrent()
{
	Instrument::ThreadLocalData &threadLocal = Instrument::getThreadLocalData();
	if (&threadLocal != &Instrument::getSentinelNonWorkerThreadLocalData()) {
		return threadLocal._context;
	} else {
		Instrument::ExternalThreadLocalData &externalThreadLocal = Instrument::getExternalThreadLocalData();
		return externalThreadLocal._context;
	}
}

void Instrument::ThreadInstrumentationContext::updateComputePlace(compute_place_id_t const &computePlaceId)
{
	Instrument::ThreadLocalData &threadLocal = Instrument::getThreadLocalData();
	
	threadLocal._context._computePlaceId = computePlaceId;
}

