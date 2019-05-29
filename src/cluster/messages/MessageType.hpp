/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef MESSAGE_TYPE_HPP
#define MESSAGE_TYPE_HPP

//! Maximum length of the name of a message
#define MSG_NAMELEN 32

typedef enum {
	SYS_FINISH = 0,
	DATA_RAW,
	DMALLOC,
	DFREE,
	DATA_FETCH,
	DATA_SEND,
	TASK_NEW,
	TASK_FINISHED,
	SATISFIABILITY,
	RELEASE_ACCESS,
	TOTAL_MESSAGE_TYPES
} MessageType;

//! Defined in MessageType.cpp
extern const char MessageTypeStr[TOTAL_MESSAGE_TYPES][MSG_NAMELEN];

#endif /* MESSAGE_TYPE_HPP */
