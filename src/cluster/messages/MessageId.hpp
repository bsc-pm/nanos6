/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_ID_HPP
#define MESSAGE_ID_HPP

#include <atomic>

namespace MessageId {
	
	//! \brief Get the next available MessageId
	uint16_t nextMessageId();
}

#endif /* MESSAGE_ID_HPP */
