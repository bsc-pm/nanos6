/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023-2024 Barcelona Supercomputing Center (BSC)
*/

#ifndef DIRECTORY_PAGE_HPP
#define DIRECTORY_PAGE_HPP

#include <pthread.h>

#include "lowlevel/SpinLock.hpp"
#include "scheduling/Scheduler.hpp"
#include "support/Containers.hpp"

enum DirectoryPageState {
	StateInvalid = 0,
	StateExclusive,
	StateShared,
	StateModified,
	StateTransitionExclusive,
	StateTransitionShared,
	StateTransitionModified
};

class Task;

struct DirectoryPageAgentInfo {
	DirectoryPageAgentInfo() :
		_state(StateInvalid),
		_allocation(nullptr),
		_copyHandler(nullptr)
	{
	}

	DirectoryPageState _state;
	void *_allocation;
	void *_copyHandler;
	Container::vector<Task *> _pendingNotifications;
};

struct DirectoryPage {
private:
	static inline DirectoryPageState finishTransition(DirectoryPageState oldState)
	{
		switch (oldState) {
			case StateTransitionShared:
				return StateShared;
			case StateTransitionModified:
				return StateModified;
			case StateTransitionExclusive:
				return StateExclusive;
			default:
				FatalErrorHandler::fail("Invalid State Transition");
				// Otherwise GCC complains
				return StateExclusive;
		}
	}
public:
	Container::vector<DirectoryPageAgentInfo> _agentInfo;
	SpinLock _lock;

	inline void lock()
	{
		_lock.lock();
	}

	inline void unlock()
	{
		_lock.unlock();
	}

	DirectoryPage(int maxDevices) :
		_agentInfo(maxDevices)
	{
	}

	void notifyCopyFinalization(int deviceId)
	{
		DirectoryPageAgentInfo &agentInfo = _agentInfo[deviceId];
		agentInfo._state = finishTransition(agentInfo._state);
		agentInfo._copyHandler = nullptr;

		for (Task *t : agentInfo._pendingNotifications) {
			if (t->decreasePredecessors())
				Scheduler::addReadyTask(t, nullptr, SIBLING_TASK_HINT);
		}

		agentInfo._pendingNotifications.clear();
	}
};

#endif // DIRECTORY_PAGE_HPP