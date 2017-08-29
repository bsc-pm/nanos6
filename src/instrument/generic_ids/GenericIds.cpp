/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "GenericIds.hpp"


namespace Instrument {
	namespace GenericIds {
		std::atomic<thread_id_t::inner_type_t> _nextThreadId(0);
		std::atomic<external_thread_id_t::inner_type_t> _nextExternalThreadId(0);
	}
}

