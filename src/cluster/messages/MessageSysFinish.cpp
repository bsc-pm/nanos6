/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "MessageSysFinish.hpp"

MessageSysFinish::MessageSysFinish(const ClusterNode *from)
	: Message("MessageSysFinish", SYS_FINISH, 1, from)
{}

void MessageSysFinish::handleMessage()
{
}
