/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "DefaultThreadManagerPolicy.hpp"
#include "ThreadManagerPolicy.hpp"


ThreadManagerPolicyInterface *ThreadManagerPolicy::_policy = nullptr;


void ThreadManagerPolicy::initialize()
{
	_policy = new DefaultThreadManagerPolicy();
}


#include "executors/threads/WorkerThread.hpp"
#include "executors/threads/WorkerThreadImplementation.hpp"
