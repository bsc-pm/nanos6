/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef __MESSAGE_TYPE_HPP__
#define __MESSAGE_TYPE_HPP__

//! Maximum length of the name of a message
#define MSG_NAMELEN 32

typedef enum {
	SYS_FINISH = 0,
	DATA_RAW,
	TOTAL_MESSAGE_TYPES
} MessageType;

//! Defined in MessageType.cpp
extern const char MessageTypeStr[TOTAL_MESSAGE_TYPES][MSG_NAMELEN];

#endif /* __MESSAGE_TYPE_HPP__ */
