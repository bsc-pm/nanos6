/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef NO_SLEEP_PRIORITY_SCHEDULER_HPP
#define NO_SLEEP_PRIORITY_SCHEDULER_HPP


#include <atomic>
#include <deque>
#include <queue>
#include <vector>

#include "../SchedulerInterface.hpp"
#include "lowlevel/PaddedTicketSpinLock.hpp"
#include "lowlevel/TicketSpinLock.hpp"
#include "executors/threads/CPU.hpp"


class Task;


class NoSleepPriorityScheduler: public SchedulerInterface {
	typedef PaddedTicketSpinLock<> spinlock_t;
	
	
	struct TaskPriorityCompare {
		inline bool operator()(Task *a, Task *b);
	};
	
	typedef std::priority_queue<Task *, std::vector<Task *>, TaskPriorityCompare> task_queue_t;
	typedef std::deque<polling_slot_t *> polling_slot_queue_t;
	
	spinlock_t _globalLock;
	
	task_queue_t _readyTasks;
	task_queue_t _unblockedTasks;
	polling_slot_queue_t _pollingSlots;
	
	
public:
	NoSleepPriorityScheduler(int numaNodeIndex);
	~NoSleepPriorityScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint, bool doGetIdle = true);
	
	Task *getReadyTask(ComputePlace *computePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true, bool doWait = false);
	
	ComputePlace *getIdleComputePlace(bool force=false);
	
	void disableComputePlace(ComputePlace *computePlace);
	
	bool requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle = true);
	bool releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle = true);
	
	std::string getName() const;
};


#endif // NO_SLEEP_PRIORITY_SCHEDULER_HPP

