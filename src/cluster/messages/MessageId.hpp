/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_ID_HPP__
#define MESSAGE_ID_HPP__

#include <atomic>

namespace MessageId {
	namespace {
		//! At the moment, the Message identifiers are 16 bits
		//! unsigned integers
		std::atomic<uint16_t> _nextMessageId(0);
	};
	
	//! \brief Get the next available MessageId
	uint16_t nextMessageId()
	{
		uint16_t ret = _nextMessageId++;
		assert(ret != UINT16_MAX);
		
		return ret;
	}
};


#endif /* MESSAGE_ID_HPP__ */
