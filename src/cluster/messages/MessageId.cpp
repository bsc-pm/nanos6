/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2018-2019 Barcelona Supercomputing Center (BSC)
*/

#include <atomic>
#include <cassert>

#include "MessageId.hpp"

namespace MessageId {
	
	typedef std::atomic<uint32_t> message_id_t;
	
	static message_id_t _nextMessageId[TOTAL_MESSAGE_TYPES];
	static uint32_t messageMax = (1UL << 24);
	
	uint32_t nextMessageId(MessageType type)
	{
		uint32_t ret = _nextMessageId[type]++;
		assert(ret != messageMax);
		
		return ret;
	}
}
