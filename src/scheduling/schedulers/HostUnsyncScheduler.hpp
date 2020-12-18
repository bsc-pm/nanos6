/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef HOST_UNSYNC_SCHEDULER_HPP
#define HOST_UNSYNC_SCHEDULER_HPP

#include "UnsyncScheduler.hpp"
#include "scheduling/ready-queues/DeadlineQueue.hpp"
#include "scheduling/ready-queues/ReadyQueueDeque.hpp"
#include "scheduling/ready-queues/ReadyQueueLocality.hpp"
#include "scheduling/ready-queues/ReadyQueueMap.hpp"
#include "support/Containers.hpp"

class Taskfor;

class HostUnsyncScheduler : public UnsyncScheduler {
	typedef Container::vector<Taskfor *> taskfor_group_slots_t;

	taskfor_group_slots_t _groupSlots;
	bool _enableLocality;

public:
	HostUnsyncScheduler(SchedulingPolicy policy, bool enablePriority, bool enableImmediateSuccessor, bool enableLocality) :
		UnsyncScheduler(policy, enablePriority, enableImmediateSuccessor), _enableLocality(enableLocality)
	{
		assert(!(enablePriority && enableLocality));
		assert(!enableLocality || DataTrackingSupport::isTrackingEnabled());

		if (enablePriority) {
			_readyTasks = new ReadyQueueMap(policy);
		} else if (enableLocality) {
			size_t numL2Queues = HardwareInfo::getNumL2Cache();
			size_t numL3Queues = HardwareInfo::getNumL3Cache();
			uint8_t numNUMAQueues = HardwareInfo::getValidMemoryPlaceCount(nanos6_host_device);
			_readyTasks = new ReadyQueueLocality(policy, numL2Queues, numL3Queues, numNUMAQueues);
		} else {
			_readyTasks = new ReadyQueueDeque(policy);
		}

		size_t groups = CPUManager::getNumTaskforGroups();

		_groupSlots = taskfor_group_slots_t(groups, nullptr);

		if (enableImmediateSuccessor && !enableLocality) {
			_immediateSuccessorTaskfors = immediate_successor_tasks_t(groups*2, nullptr);
		}
		if (enableImmediateSuccessor && !enableLocality) {
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

	//! \brief Add a (ready) task that has been created or freed
	//!
	//! \param[in] task the task to be added
	//! \param[in] computePlace the hardware place of the creator or the liberator
	//! \param[in] hint a hint about the relation of the task to the current task
	virtual inline void addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint = NO_HINT)
	{
		assert(task != nullptr);

		bool unblocked = (hint == UNBLOCKED_TASK_HINT);

		if (!_enableLocality) {
			if (_enableImmediateSuccessor) {
				if (computePlace != nullptr && hint == SIBLING_TASK_HINT) {
					size_t immediateSuccessorId = computePlace->getIndex();
					if (!task->isTaskfor()) {
						Task *currentIS = _immediateSuccessorTasks[immediateSuccessorId];
						if (currentIS != nullptr) {
							assert(!currentIS->isTaskfor());
							_readyTasks->addReadyTask(currentIS, false);
						}
						_immediateSuccessorTasks[immediateSuccessorId] = task;
					} else {
						// Multiply by 2 because there are 2 slots per group
						immediateSuccessorId = ((CPU *)computePlace)->getGroupId()*2;
						Task *currentIS1 = _immediateSuccessorTaskfors[immediateSuccessorId];
						Task *currentIS2 = _immediateSuccessorTaskfors[immediateSuccessorId+1];
						if (currentIS1 == nullptr) {
							_immediateSuccessorTaskfors[immediateSuccessorId] = task;
						}
						else if (currentIS2 == nullptr) {
							_immediateSuccessorTaskfors[immediateSuccessorId+1] = task;
						}
						else {
							_readyTasks->addReadyTask(currentIS1, false);
							_immediateSuccessorTaskfors[immediateSuccessorId] = task;
						}
					}
					return;
				}
			}
		}

		_readyTasks->addReadyTask(task, unblocked);
	}

	//! \brief Get a ready task for execution
	//!
	//! \param[in] computePlace The hardware place asking for scheduling orders
	//!
	//! \returns A ready task or nullptr
	Task *getReadyTask(ComputePlace *computePlace);
};

#endif // HOST_UNSYNC_SCHEDULER_HPP
