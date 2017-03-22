#ifndef LOCALITY_SCHEDULER_HPP
#define LOCALITY_SCHEDULER_HPP


#include <deque>
#include <vector>
#include <unordered_map>

#include "SchedulerInterface.hpp"
#include "lowlevel/SpinLock.hpp"
#include "executors/threads/CPU.hpp"
#include "instrument/stats/Timer.hpp"


class Task;


class LocalityScheduler: public SchedulerInterface {
    //! Members for debugging purposes
    Instrument::Timer _timerWaits;

	SpinLock _globalLock;
	
    //! Tasks with logical dependences satisfied but data is not in the remote host.
	//std::deque<Task *> _preReadyTasks;
    //! Tasks ready to be executed.
    //! There must be a queue for each NUMA node. Also, there must be a lock for queue.
    std::deque<Task *> **_readyQueues;
    std::size_t _readyQueuesSize;
    //std::unordered_map<unsigned int, std::deque<Task *> > _readyQueues;
	//!std::deque<Task *> _readyTasks;
	std::deque<Task *> _unblockedTasks;
	
    inline CPU *getLocalityCPU(Task * task);
	inline Task *getReplacementTask(CPU *hardwarePlace);
	
public:
	LocalityScheduler();
	~LocalityScheduler();
	
	ComputePlace *addReadyTask(Task *task, ComputePlace *hardwarePlace, ReadyTaskHint hint);
	
	void taskGetsUnblocked(Task *unblockedTask, ComputePlace *hardwarePlace);
	
	Task *getReadyTask(ComputePlace *hardwarePlace, Task *currentTask = nullptr);
	
	ComputePlace *getIdleComputePlace(bool force=false);

    void createReadyQueues(std::size_t nodes);
};


#endif // LOCALITY_SCHEDULER_HPP

