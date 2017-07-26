/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENTED_THREAD_HPP
#define INSTRUMENTED_THREAD_HPP


#include <InstrumentThreadId.hpp>


class InstrumentedThread {
protected:
	Instrument::thread_id_t _instrumentationId;
	
	
public:
	Instrument::thread_id_t getInstrumentationId() const
	{
		return _instrumentationId;
	}
	
	void setInstrumentationId(Instrument::thread_id_t const &instrumentationId)
	{
		_instrumentationId = instrumentationId;
	}
};


#endif // INSTRUMENTED_THREAD_HPP
