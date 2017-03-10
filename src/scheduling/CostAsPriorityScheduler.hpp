#ifndef COST_AS_PRIORITY_SCHEDULER_HPP
#define COST_AS_PRIORITY_SCHEDULER_HPP


#include <atomic>
#include <deque>
#include <queue>
#include <vector>

#include "SchedulerInterface.hpp"
#include "lowlevel/PaddedTicketSpinLock.hpp"
#include "lowlevel/TicketSpinLock.hpp"
#include "executors/threads/CPU.hpp"


class Task;


class CostAsPriorityScheduler: public SchedulerInterface {
	typedef PaddedTicketSpinLock<> spinlock_t;
	
	struct PriorityCompare {
		inline bool operator()(Task *a, Task *b);
	};
	
	spinlock_t _globalLock;
	
	std::priority_queue<Task *, std::vector<Task *>, PriorityCompare> _readyTasks;
	std::priority_queue<Task *, std::vector<Task *>, PriorityCompare> _unblockedTasks;
	
	std::deque<CPU *> _idleCPUs;
	
	std::atomic<std::atomic<Task *> *> _pollingSlot;
	
	
	inline CPU *getIdleCPU();
	inline void cpuBecomesIdle(CPU *cpu);
	
public:
	CostAsPriorityScheduler();
	~CostAsPriorityScheduler();
	
	HardwarePlace *addReadyTask(Task *task, HardwarePlace *hardwarePlace, ReadyTaskHint hint);
	
	void taskGetsUnblocked(Task *unblockedTask, HardwarePlace *hardwarePlace);
	
	Task *getReadyTask(HardwarePlace *hardwarePlace, Task *currentTask = nullptr);
	
	HardwarePlace *getIdleHardwarePlace(bool force=false);
	
	void disableHardwarePlace(HardwarePlace *hardwarePlace);
	
	bool requestPolling(HardwarePlace *hardwarePlace, std::atomic<Task *> *pollingSlot);
	bool releasePolling(HardwarePlace *hardwarePlace, std::atomic<Task *> *pollingSlot);
};


#endif // COST_AS_PRIORITY_SCHEDULER_HPP

