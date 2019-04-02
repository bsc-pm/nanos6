/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/


#ifndef EXTERNAL_THREAD_GROUP_HPP
#define EXTERNAL_THREAD_GROUP_HPP


#include <mutex>
#include <vector>

#include "ExternalThread.hpp"


class ExternalThreadGroup {
private:
	typedef PaddedTicketSpinLock<int, 128> spinlock_t;
	
	//! Static variable that points to the global external thread group
	static ExternalThreadGroup *_externalThreadGroup;
	
	std::vector<ExternalThread *> _externalThreads;
	spinlock_t _lock;
	
	ExternalThreadGroup()
		: _externalThreads(), _lock()
	{
	}
	
public:
	//! \brief Initialize the ExternalThreadGroup as empty
	static inline void initialize()
	{
		_externalThreadGroup = new ExternalThreadGroup();
		assert(_externalThreadGroup != nullptr);
	}
	
	//! \brief Shutdown the ExternalThreadGroup
	//! 
	//! This deletes all ExternalThreads that were registered
	//! in the group.
	static inline void shutdown()
	{
		assert(_externalThreadGroup != nullptr);
		
		_externalThreadGroup->deleteAll();
		
		delete _externalThreadGroup;
	}
	
	//! \brief Register an existing ExternalThread in the group
	//! 
	//! This registers the existing externalThread in the group. Note
	//! that the group must be already initialized when calling this
	//! function.
	//! 
	//! \param[in] externalThread is the external thread to register
	static inline void registerExternalThread(ExternalThread *externalThread)
	{
		assert(externalThread != nullptr);
		assert(_externalThreadGroup != nullptr);
		
		_externalThreadGroup->add(externalThread);
	}
	
private:
	inline void add(ExternalThread *externalThread)
	{
		std::lock_guard<spinlock_t> guard(_lock);
		_externalThreads.push_back(externalThread);
	}
	
	inline void deleteAll()
	{
		std::lock_guard<spinlock_t> guard(_lock);
		
		// Delete all registered external threads
		for (ExternalThread *thread : _externalThreads) {
			assert(thread != nullptr);
			delete thread;
		}
	}
};


#endif // EXTERNAL_THREAD_GROUP_HPP

