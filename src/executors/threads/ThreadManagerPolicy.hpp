/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef THREAD_MANAGER_POLICY_HPP
#define THREAD_MANAGER_POLICY_HPP

#include <cassert>

#include "CPUActivation.hpp"
#include "tasks/Task.hpp"

class ThreadManagerPolicy {
public:
	enum thread_run_inline_policy_t {
		POLICY_NO_INLINE = 0,
		POLICY_CHILDREN_INLINE,
		POLICY_ALL_INLINE
	};

	//! \brief check if a task must be run within the thread of a blocked task
	//!
	//! \param[in] replacementTask the task that is to be run
	//! \param[in] currentTask the task that is blocked on the thread
	//! \param[in] cpu the CPU where currentTask is running
	//!
	//! \returns true is replacementTask is to be run within the same thread as currentTask (which is blocked)
	static bool checkIfMustRunInline(Task *replacementTask, Task *currentTask, CPU *cpu, thread_run_inline_policy_t policy)
	{
		assert(replacementTask != nullptr);
		assert(currentTask != nullptr);
		assert(cpu != nullptr);
		
		if (replacementTask->getThread() != nullptr) {
			return false;
		}
		
		if (!CPUActivation::acceptsWork(cpu)) {
			return false;
		}
		
		bool mustRunInline = false;
		switch (policy) {
			case POLICY_NO_INLINE:
				mustRunInline = false;
				break;
			case POLICY_CHILDREN_INLINE:
				mustRunInline = replacementTask->getParent() == currentTask;
				break;
			case POLICY_ALL_INLINE:
				mustRunInline = true;
				break;
		}
		
		return mustRunInline;
	}
	
	
	//! \brief check if a task that has just been unblocked must preempt its unblocker
	//!
	//! \param[in] unblockerTask the task that unblocks the other task (the triggerer)
	//! \param[in] unblockedTask the task that is unblocked by the other task
	//! \param[in] cpu the CPU where unblockerTask is running
	//!
	//! \returns true if unblockedTask must preempt unblockerTask
	static bool checkIfUnblockedMustPreemtUnblocker(
		__attribute__((unused)) Task *unblockerTask,
		__attribute__((unused)) Task *unblockedTask,
		__attribute__((unused)) CPU *cpu)
	{
		assert(unblockerTask != nullptr);
		assert(unblockedTask != nullptr);
		assert(cpu != nullptr);
		
		return true;
	}
	
};


#endif // THREAD_MANAGER_POLICY_HPP
