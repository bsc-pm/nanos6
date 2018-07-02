/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "KernelLevelThread.hpp"


__thread KernelLevelThread *KernelLevelThread::_currentKernelLevelThread(nullptr);

#ifndef NDEBUG
namespace ompss_debug {
	void *getCurrentThread()
	{
		return KernelLevelThread::getCurrentKernelLevelThread();
	}
}
#endif
