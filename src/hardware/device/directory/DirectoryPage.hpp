/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef DIRECTORY_PAGE_HPP
#define DIRECTORY_PAGE_HPP

#include <pthread.h>

#include "lowlevel/SpinLock.hpp"
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

struct DirectoryPage {
	Container::vector<DirectoryPageState> _states;
	Container::vector<void *> _allocations;
	Container::vector<void *> _copyHandlers;
	Container::vector<Container::vector<Task *>> _pendingNotifications;
	pthread_spinlock_t _lock;
	// SpinLock _lock;

	inline void lock()
	{
		pthread_spin_lock(&_lock);
	}

	inline void unlock()
	{
		pthread_spin_unlock(&_lock);
	}

	DirectoryPage(int maxDevices) :
		_states(maxDevices),
		_allocations(maxDevices),
		_copyHandlers(maxDevices),
		_pendingNotifications(maxDevices)
	{
		pthread_spin_init(&_lock, 0);
	}

	~DirectoryPage() {
		pthread_spin_destroy(&_lock);
	}
};

#endif // DIRECTORY_PAGE_HPP