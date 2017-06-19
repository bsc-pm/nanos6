#ifndef FIFO_IMMEDIATE_SUCCESSOR_WITH_MULTI_POLLING_SCHEDULER_HPP
#define FIFO_IMMEDIATE_SUCCESSOR_WITH_MULTI_POLLING_SCHEDULER_HPP


#include <atomic>
#include <deque>
#include <vector>

#include "../SchedulerInterface.hpp"
#include "lowlevel/PaddedTicketSpinLock.hpp"
#include "executors/threads/CPU.hpp"


class Task;


class FIFOImmediateSuccessorWithMultiPollingScheduler: public SchedulerInterface {
	typedef PaddedTicketSpinLock<> spinlock_t;
	
	spinlock_t _globalLock;
	
	std::deque<Task *> _readyTasks;
	std::deque<Task *> _unblockedTasks;
	
	std::atomic<polling_slot_t *> _pollingSlot;
	
	
	inline Task *getReplacementTask(CPU *computePlace);
	
public:
	FIFOImmediateSuccessorWithMultiPollingScheduler(int numaNodeIndex);
	~FIFOImmediateSuccessorWithMultiPollingScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint, bool doGetIdle = true);
	
	void taskGetsUnblocked(Task *unblockedTask, ComputePlace *computePlace);
	
	Task *getReadyTask(ComputePlace *computePlace, Task *currentTask = nullptr, bool canMarkAsIdle = true);
	
	ComputePlace *getIdleComputePlace(bool force=false);
	
	void disableComputePlace(ComputePlace *computePlace);
	
	bool requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot);
	bool releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot);
	
	std::string getName() const;
};


#endif // FIFO_IMMEDIATE_SUCCESSOR_WITH_MULTI_POLLING_SCHEDULER_HPP

