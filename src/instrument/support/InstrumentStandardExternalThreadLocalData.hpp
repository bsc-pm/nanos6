/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_STANDARD_EXTERNAL_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_STANDARD_EXTERNAL_THREAD_LOCAL_DATA_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentExternalThreadId.hpp>

#include <string>


namespace Instrument {
	struct StandardExternalThreadLocalData {
		external_thread_id_t _currentThreadId;
		std::string _name;
		
		InstrumentationContext _context;
		
		StandardExternalThreadLocalData(std::string const &externalThreadName)
			: _currentThreadId(), _name(externalThreadName), _context(&_name)
		{
		}
	};
}


#endif // INSTRUMENT_STANDARD_EXTERNAL_THREAD_LOCAL_DATA_HPP
