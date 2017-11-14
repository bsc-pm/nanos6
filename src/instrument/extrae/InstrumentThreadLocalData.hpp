/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_EXTRAE_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_EXTRAE_THREAD_LOCAL_DATA_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <instrument/support/sampling/ThreadLocalData.hpp>

#include "InstrumentThreadId.hpp"

#include <set>
#include <vector>


namespace Instrument {
	struct ThreadLocalData : public Instrument::Sampling::ThreadLocalData {
		enum {
			max_backlog = 4096
		};
		
		thread_id_t _currentThreadId;
		std::vector<int> _nestingLevels;
		std::set<void *> _backtraceAddresses;
		
		int _inMemoryAllocation;
		std::vector<void *> _backtraceAddressBacklog;
		
		InstrumentationContext _context;
		
		ThreadLocalData()
			: _currentThreadId(), _nestingLevels(),
			_backtraceAddresses(),
			_inMemoryAllocation(0), _backtraceAddressBacklog()
		{
		}
		
		// NOTE: this is not in the constructor because the constructor is called on the first access to the TLS
		// but before the TLS has settled, which whould cause the memory allocation inteception to attempt to
		// retrieve it, failing, causing a call to the costructor, ... and this an infinite recursion.
		inline void init()
		{
			_backtraceAddressBacklog.reserve(max_backlog);
		}
	};
}


#endif // INSTRUMENT_EXTRAE_THREAD_LOCAL_DATA_HPP
