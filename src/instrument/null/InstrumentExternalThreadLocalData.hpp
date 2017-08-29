/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_EXTERNAL_THREAD_LOCAL_DATA_HPP
#define INSTRUMENT_NULL_EXTERNAL_THREAD_LOCAL_DATA_HPP


#include <string>


namespace Instrument {
	struct ExternalThreadLocalData {
		ExternalThreadLocalData(__attribute__((unused)) std::string const &externalThreadName)
		{
		}
	};
}


#endif // INSTRUMENT_NULL_EXTERNAL_THREAD_LOCAL_DATA_HPP
