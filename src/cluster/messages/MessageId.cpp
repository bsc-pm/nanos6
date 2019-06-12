/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018-2019 Barcelona Supercomputing Center (BSC)
*/

#include <atomic>
#include <cassert>

#include "MessageId.hpp"


namespace MessageId {
	
	static std::atomic<uint16_t> _nextMessageId(0);
	
	uint16_t nextMessageId()
	{
		uint16_t ret = _nextMessageId++;
		assert(ret != UINT16_MAX);
		
		return ret;
	}
}
