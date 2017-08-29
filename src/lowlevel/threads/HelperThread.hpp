/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/


#ifndef HELPER_THREAD_HPP
#define HELPER_THREAD_HPP

#include "ExternalThread.hpp"
#include "KernelLevelThread.hpp"


class HelperThread : public ExternalThread, public KernelLevelThread {
protected:
	inline void initializeHelperThread()
	{
		initializeExternalThread();
	}
	
public:
	template<typename... TS>
	HelperThread(TS... nameComponents)
		: ExternalThread(nameComponents...), KernelLevelThread()
	{
	}
	
};


#endif // HELPER_THREAD_HPP

