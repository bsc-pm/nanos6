/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_DEPENDENCY_DATA_HPP
#define CPU_DEPENDENCY_DATA_HPP


#include <atomic>
#include <cassert>
#include <deque>
#include <queue>

#include <limits.h>

#include "DataAccessFlags.hpp"

class Task;

struct CPUDependencyData {
	typedef std::deque<Task *> satisfied_originator_list_t;
	typedef std::deque<Task *> deletable_originator_list_t;

	//! Tasks whose accesses have been satisfied after ending a task
	satisfied_originator_list_t _satisfiedOriginators;
	deletable_originator_list_t _deletableOriginators;
	mailbox_t _mailBox;

	CPUDependencyData()
		: _satisfiedOriginators(),
		_deletableOriginators(),
		_mailBox()
	{
	}

	~CPUDependencyData()
	{
		assert(empty());
	}

	inline bool empty() const
	{
		return _satisfiedOriginators.empty() && _deletableOriginators.empty() && _mailBox.empty();
	}
};


#endif // CPU_DEPENDENCY_DATA_HPP
