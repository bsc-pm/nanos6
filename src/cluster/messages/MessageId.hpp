/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_ID_HPP
#define MESSAGE_ID_HPP

#include <atomic>

namespace MessageId {
	
	//! \brief Get the next available MessageId
	uint16_t nextMessageId()
	{
		static std::atomic<uint16_t> _nextMessageId(0);
		
		uint16_t ret = _nextMessageId++;
		assert(ret != UINT16_MAX);
		
		return ret;
	}
}

#endif /* MESSAGE_ID_HPP */
