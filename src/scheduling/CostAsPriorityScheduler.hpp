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
	
	
	
	
	struct TaskPriorityCompare {
		inline bool operator()(Task *a, Task *b);
	};
	
	typedef std::priority_queue<Task *, std::vector<Task *>, TaskPriorityCompare> task_queue_t;
	
	
	spinlock_t _globalLock;
	
	task_queue_t _readyTasks;
	task_queue_t _unblockedTasks;
	
	std::deque<CPU *> _idleCPUs;
	
	std::atomic<polling_slot_t *> _pollingSlot;
	
	
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
	
	bool requestPolling(HardwarePlace *hardwarePlace, polling_slot_t *pollingSlot);
	bool releasePolling(HardwarePlace *hardwarePlace, polling_slot_t *pollingSlot);
};


#endif // COST_AS_PRIORITY_SCHEDULER_HPP

