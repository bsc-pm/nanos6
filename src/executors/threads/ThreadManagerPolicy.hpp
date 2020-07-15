/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef THREAD_MANAGER_POLICY_HPP
#define THREAD_MANAGER_POLICY_HPP

#include <cassert>

#include "tasks/Task.hpp"


class ThreadManagerPolicy {

public:

	//! \brief Check if a task that has just been unblocked must preempt its unblocker
	//!
	//! \param[in] unblockerTask The task that unblocks the other task (the triggerer)
	//! \param[in] unblockedTask The task that is unblocked by the other task
	//! \param[in] cpu The CPU where unblockerTask is running
	//!
	//! \returns True if unblockedTask must preempt unblockerTask
	static inline bool checkIfUnblockedMustPreemptUnblocker(
		__attribute__((unused)) Task *unblockerTask,
		__attribute__((unused)) Task *unblockedTask,
		__attribute__((unused)) CPU *cpu
	) {
		assert(unblockerTask != nullptr);
		assert(unblockedTask != nullptr);
		assert(cpu != nullptr);

		// Do not preempt the unblocker task. Preempting the unblockers
		// tends to worsen the execution time in general. TODO: We should
		// decide whether it is better to dynamically enable/disable the
		// preemption depending on the current conditions or just avoid
		// preemption in all cases
		return false;
	}

};

#endif // THREAD_MANAGER_POLICY_HPP
