/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_SAMPLING_SIGPROF_HPP
#define INSTRUMENT_SUPPORT_SAMPLING_SIGPROF_HPP


#include "ThreadLocalData.hpp"

#include <instrument/support/InstrumentThreadLocalDataSupport.hpp>


namespace Instrument {
namespace Sampling {


class SigProf {
public:
	typedef void (*handler_t)(Instrument::Sampling::ThreadLocalData &);
	
private:
	static bool _enabled;
	static unsigned int _nsPerSample;
	static handler_t _handler;
	
public:
	static void init();
	
	static void setPeriod(unsigned int nsPerSample)
	{
		_nsPerSample = nsPerSample;
	}
	static void setHandler(handler_t handler)
	{
		_handler = handler;
	}
	static handler_t getHandler()
	{
		return _handler;
	}
	
	static void enable()
	{
		_enabled = true;
	}
	static void disable()
	{
		_enabled = false;
	}
	static inline bool isEnabled()
	{
		return _enabled;
	}
	
	static void setUpThread(Instrument::Sampling::ThreadLocalData &threadLocal = getThreadLocalData());
	static void enableThread(Instrument::Sampling::ThreadLocalData &threadLocal = getThreadLocalData());
	static void disableThread(Instrument::Sampling::ThreadLocalData &threadLocal = getThreadLocalData());
	
	// Returns true is this is the call that actually reenables it
	static bool lightweightEnableThread(Instrument::Sampling::ThreadLocalData &threadLocal = getThreadLocalData());
	
	// Returns true if this is the call that actually disables it
	static bool lightweightDisableThread(Instrument::Sampling::ThreadLocalData &threadLocal = getThreadLocalData());
	
	static void forceHandler();
};


} // namespace Sampling
} // namespace Instrument


#endif // INSTRUMENT_SUPPORT_SAMPLING_SIGPROF_HPP
