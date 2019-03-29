/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef FIFO_IMMEDIATE_SUCCESSOR_WITH_POLLING_SCHEDULER_HPP
#define FIFO_IMMEDIATE_SUCCESSOR_WITH_POLLING_SCHEDULER_HPP


#include <atomic>
#include <deque>
#include <vector>

#include "../SchedulerInterface.hpp"
#include "lowlevel/PaddedTicketSpinLock.hpp"
#include "lowlevel/TicketSpinLock.hpp"
#include "executors/threads/CPU.hpp"


class Task;


class FIFOImmediateSuccessorWithPollingScheduler: public SchedulerInterface {
	typedef PaddedTicketSpinLock<> spinlock_t;
	
	spinlock_t _globalLock;
	
	std::deque<Task *> _readyTasks;
	std::deque<Task *> _unblockedTasks;
	
	std::atomic<polling_slot_t *> _pollingSlot;
	
	
	inline Task *getReplacementTask(CPU *computePlace);
	
public:
	FIFOImmediateSuccessorWithPollingScheduler(int numaNodeIndex);
	~FIFOImmediateSuccessorWithPollingScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint, bool doGetIdle = true);
	
	Task *getReadyTask(ComputePlace *computePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true, bool doWait = false);
	
	ComputePlace *getIdleComputePlace(bool force=false);
	
	void disableComputePlace(ComputePlace *computePlace);
	
	bool requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle = true);
	bool releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot, bool canMarkAsIdle = true);
	
	std::string getName() const;
};


#endif // FIFO_IMMEDIATE_SUCCESSOR_WITH_POLLING_SCHEDULER_HPP

