#ifndef IMMEDIATE_SUCCESSOR_SCHEDULER_HPP
#define IMMEDIATE_SUCCESSOR_SCHEDULER_HPP


#include <deque>
#include <vector>

#include "SchedulerInterface.hpp"
#include "lowlevel/SpinLock.hpp"
#include "executors/threads/CPU.hpp"


class Task;


class ImmediateSuccessorScheduler: public SchedulerInterface {
	SpinLock _globalLock;
	
	std::deque<Task *> _readyTasks;
	std::deque<Task *> _unblockedTasks;
	
	std::deque<CPU *> _idleCPUs;
	
	inline CPU *getIdleCPU();
	inline Task *getReplacementTask(CPU *hardwarePlace);
	inline void cpuBecomesIdle(CPU *cpu);
	
public:
	ImmediateSuccessorScheduler();
	~ImmediateSuccessorScheduler();
	
	HardwarePlace *addReadyTask(Task *task, HardwarePlace *hardwarePlace, ReadyTaskHint hint);
	
	void taskGetsUnblocked(Task *unblockedTask, HardwarePlace *hardwarePlace);
	
	bool checkIfIdleAndGrantReactivation(HardwarePlace *hardwarePlace);
	
	Task *getReadyTask(HardwarePlace *hardwarePlace, Task *currentTask = nullptr);
	
	HardwarePlace *getIdleHardwarePlace(bool force=false);
	
	void disableHardwarePlace(HardwarePlace *hardwarePlace);
};


#endif // IMMEDIATE_SUCCESSOR_SCHEDULER_HPP

