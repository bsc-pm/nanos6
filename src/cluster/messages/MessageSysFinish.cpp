/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "MessageSysFinish.hpp"
#include "cluster/ClusterManager.hpp"

#include <nanos6/bootstrap.h>

MessageSysFinish::MessageSysFinish(const ClusterNode *from)
	: Message("MessageSysFinish", SYS_FINISH, 1, from)
{}

bool MessageSysFinish::handleMessage()
{
	if (!nanos6_can_run_main()) {
		ClusterManager::ShutdownCallback *callback =
			ClusterManager::getShutdownCallback();
		
		//! We need to call the main callback.
		while (callback == nullptr) {
			//! We will spin to avoid the (not very likely) case that the
			//! Callback has not been set yet. This could happen if we
			//! received and handled a MessageSysFinish before the loader
			//! code has finished setting up everything.
			callback = ClusterManager::getShutdownCallback();
		}
		
		callback->invoke();
	}
	
	//! Synchronize with all other cluster nodes at this point
	ClusterManager::synchronizeAll();
	
	return true;
}
