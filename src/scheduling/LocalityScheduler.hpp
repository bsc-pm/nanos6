#ifndef LOCALITY_SCHEDULER_HPP
#define LOCALITY_SCHEDULER_HPP


#include <deque>
#include <vector>

#include "SchedulerInterface.hpp"
#include "lowlevel/SpinLock.hpp"
#include "executors/threads/CPU.hpp"


class Task;


class LocalityScheduler: public SchedulerInterface {
	SpinLock _globalLock;
	
    //! Tasks with logical dependences satisfied but data is not in the remote host.
	//std::deque<Task *> _preReadyTasks;
    //! Tasks ready to be executed.
	std::deque<Task *> _readyTasks;
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
};


#endif // LOCALITY_SCHEDULER_HPP

