#ifndef NAIVE_SCHEDULER_HPP
#define NAIVE_SCHEDULER_HPP


#include <deque>
#include <vector>

#include "SchedulerInterface.hpp"
#include "lowlevel/SpinLock.hpp"


class Task;
class WorkerThread;


class NaiveScheduler: public SchedulerInterface {
	SpinLock _readyTasksLock;
	std::deque<Task *> _readyTasks;
	
	
public:
	NaiveScheduler();
	~NaiveScheduler();
	
	void addMainTask(Task *mainTask);
	void addSiblingTask(Task *newReadyTask, Task *triggererTask, HardwarePlace const *hardwarePlace);
	void addChildTask(Task *newReadyTask, HardwarePlace const *hardwarePlace);
	Task *schedule(const HardwarePlace *hardwarePlace);
	
};


#endif // NAIVE_SCHEDULER_HPP

