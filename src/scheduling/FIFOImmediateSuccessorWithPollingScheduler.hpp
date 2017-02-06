#ifndef FIFO_IMMEDIATE_SUCCESSOR_WITH_POLLING_SCHEDULER_HPP
#define FIFO_IMMEDIATE_SUCCESSOR_WITH_POLLING_SCHEDULER_HPP


#include <atomic>
#include <deque>
#include <vector>

#include "SchedulerInterface.hpp"
#include "lowlevel/TicketSpinLock.hpp"
#include "executors/threads/CPU.hpp"


class Task;


class FIFOImmediateSuccessorWithPollingScheduler: public SchedulerInterface {
	typedef TicketSpinLock<> spinlock_t;
	
	spinlock_t _globalLock;
	
	std::deque<Task *> _readyTasks;
	std::deque<Task *> _unblockedTasks;
	
	std::deque<CPU *> _idleCPUs;
	
	std::atomic<std::atomic<Task *> *> _pollingSlot;
	
	
	inline CPU *getIdleCPU();
	inline Task *getReplacementTask(CPU *hardwarePlace);
	inline void cpuBecomesIdle(CPU *cpu);
	
public:
	FIFOImmediateSuccessorWithPollingScheduler();
	~FIFOImmediateSuccessorWithPollingScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint);
	
	void taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace);
	
	Task *getReadyTask(ComputePlace *hardwarePlace, Task *currentTask = nullptr);
	
	ComputePlace *getIdleComputePlace(bool force=false);
	
	void disableComputePlace(ComputePlace *hardwarePlace);
	
	bool requestPolling(ComputePlace *hardwarePlace, std::atomic<Task *> *pollingSlot);
	bool releasePolling(ComputePlace *hardwarePlace, std::atomic<Task *> *pollingSlot);

    void addReadyQueue(std::size_t node_id);
};


#endif // FIFO_IMMEDIATE_SUCCESSOR_WITH_POLLING_SCHEDULER_HPP

