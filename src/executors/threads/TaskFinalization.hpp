#ifndef TASK_FINALIZATION_HPP
#define TASK_FINALIZATION_HPP

#include "InstrumentTaskExecution.hpp"

#include "CPU.hpp"
#include "WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/ompss/SpawnFunction.hpp"
#include "tasks/Task.hpp"



class TaskFinalization {
public:
	static void disposeOrUnblockTask(Task *task, CPU *cpu, WorkerThread *thread);
};


#endif // TASK_FINALIZATION_HPP
