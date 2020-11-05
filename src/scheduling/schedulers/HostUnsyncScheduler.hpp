/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef HOST_UNSYNC_SCHEDULER_HPP
#define HOST_UNSYNC_SCHEDULER_HPP

#include "MemoryAllocator.hpp"
#include "UnsyncScheduler.hpp"
#include "scheduling/ready-queues/DeadlineQueue.hpp"

class Taskfor;

class HostUnsyncScheduler : public UnsyncScheduler {
	typedef std::vector<Taskfor *, TemplateAllocator<Taskfor *>> taskfor_group_slots_t;
	taskfor_group_slots_t _groupSlots;

public:
	HostUnsyncScheduler(SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor)
		: UnsyncScheduler(policy, enablePriority, enableImmediateSuccessor)
	{
		size_t groups = CPUManager::getNumTaskforGroups();

		_groupSlots = taskfor_group_slots_t(groups, nullptr);

		if (enableImmediateSuccessor) {
			_immediateSuccessorTaskfors = immediate_successor_tasks_t(groups*2, nullptr);
		}

		_deadlineTasks = new DeadlineQueue(policy);
		assert(_deadlineTasks != nullptr);
	}

	virtual ~HostUnsyncScheduler()
	{
		assert(_deadlineTasks != nullptr);
		delete _deadlineTasks;
	}

	//! \brief Get a ready task for execution
	//!
	//! \param[in] computePlace The hardware place asking for scheduling orders
	//!
	//! \returns A ready task or nullptr
	Task *getReadyTask(ComputePlace *computePlace);
};

#endif // HOST_UNSYNC_SCHEDULER_HPP
