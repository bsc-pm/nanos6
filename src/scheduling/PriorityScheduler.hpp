#ifndef PRIORITY_SCHEDULER_HPP
#define PRIORITY_SCHEDULER_HPP


#include <atomic>
#include <deque>
#include <queue>
#include <vector>

#include "SchedulerInterface.hpp"
#include "lowlevel/PaddedTicketSpinLock.hpp"
#include "lowlevel/TicketSpinLock.hpp"
#include "executors/threads/CPU.hpp"


class Task;


class PriorityScheduler: public SchedulerInterface {
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
	PriorityScheduler();
	~PriorityScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *computePlace, ReadyTaskHint hint);
	
	void taskGetsUnblocked(Task *unblockedTask, ComputePlace *computePlace);
	
	Task *getReadyTask(ComputePlace *computePlace, Task *currentTask = nullptr);
	
	ComputePlace *getIdleComputePlace(bool force=false);
	
	void disableComputePlace(ComputePlace *computePlace);
	
	bool requestPolling(ComputePlace *computePlace, polling_slot_t *pollingSlot);
	bool releasePolling(ComputePlace *computePlace, polling_slot_t *pollingSlot);
};


#endif // PRIORITY_SCHEDULER_HPP

