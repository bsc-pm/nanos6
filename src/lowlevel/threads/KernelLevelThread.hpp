/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef KERNEL_LEVEL_THREAD_HPP
#define KERNEL_LEVEL_THREAD_HPP


#include "posix/KernelLevelThread.hpp"


#ifndef NDEBUG
namespace ompss_debug {
	void *getCurrentThread();
}
#endif


#endif // KERNEL_LEVEL_THREAD_HPP
