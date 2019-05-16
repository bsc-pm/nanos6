/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/


#include "ExecutionSteps.hpp"
#include "InstrumentGraph.hpp"
#include "InstrumentTaskId.hpp"
#include "InstrumentUserMutex.hpp"
#include "system/ompss/UserMutex.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <InstrumentInstrumentationContext.hpp>


namespace Instrument {
	using namespace Graph;
	
	static inline usermutex_id_t getUserMutexId(UserMutex *userMutex, __attribute__((unused)) std::lock_guard<SpinLock> const &guard)
	{
		usermutex_id_t usermutexId;
		
		usermutex_to_id_map_t::iterator it = _usermutexToId.find(userMutex);
		if (it != _usermutexToId.end()) {
			usermutexId = it->second;
		} else {
			usermutexId = _nextUsermutexId++;
			_usermutexToId[userMutex] = usermutexId;
		}
		
		return usermutexId;
	}
	
	void acquiredUserMutex(UserMutex *userMutex, InstrumentationContext const &context)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		
		usermutex_id_t usermutexId = getUserMutexId(userMutex, guard);
		
		enter_usermutex_step_t *enterUsermutexStep = new enter_usermutex_step_t(context, usermutexId);
		_executionSequence.push_back(enterUsermutexStep);
	}
	
	void blockedOnUserMutex(UserMutex *userMutex, InstrumentationContext const &context)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		
		usermutex_id_t usermutexId = getUserMutexId(userMutex, guard);
		
		block_on_usermutex_step_t *blockOnUsermutexStep = new block_on_usermutex_step_t(context, usermutexId);
		_executionSequence.push_back(blockOnUsermutexStep);
	}
	
	void releasedUserMutex(UserMutex *userMutex, InstrumentationContext const &context)
	{
		std::lock_guard<SpinLock> guard(_graphLock);
		
		usermutex_id_t usermutexId = getUserMutexId(userMutex, guard);
		
		exit_usermutex_step_t *exitUsermutexStep = new exit_usermutex_step_t(context, usermutexId);
		_executionSequence.push_back(exitUsermutexStep);
	}
	
}

