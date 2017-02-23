#ifndef FIFO_IMMEDIATE_SUCCESSOR_WITH_POLLING_SCHEDULER_HPP
#define FIFO_IMMEDIATE_SUCCESSOR_WITH_POLLING_SCHEDULER_HPP


#include <atomic>
#include <deque>
#include <vector>

#include "SchedulerInterface.hpp"
#include "lowlevel/PaddedTicketSpinLock.hpp"
#include "lowlevel/TicketSpinLock.hpp"
#include "executors/threads/CPU.hpp"


class Task;


class FIFOImmediateSuccessorWithPollingScheduler: public SchedulerInterface {
	typedef PaddedTicketSpinLock<> spinlock_t;
	
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
	
	HardwarePlace *addReadyTask(Task *task, HardwarePlace *hardwarePlace, ReadyTaskHint hint);
	
	void taskGetsUnblocked(Task *unblockedTask, HardwarePlace *hardwarePlace);
	
	Task *getReadyTask(HardwarePlace *hardwarePlace, Task *currentTask = nullptr);
	
	HardwarePlace *getIdleHardwarePlace(bool force=false);
	
	void disableHardwarePlace(HardwarePlace *hardwarePlace);
	
	bool requestPolling(HardwarePlace *hardwarePlace, std::atomic<Task *> *pollingSlot);
	bool releasePolling(HardwarePlace *hardwarePlace, std::atomic<Task *> *pollingSlot);
};


#endif // FIFO_IMMEDIATE_SUCCESSOR_WITH_POLLING_SCHEDULER_HPP

