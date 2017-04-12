#ifndef TASK_FINALIZATION_HPP
#define TASK_FINALIZATION_HPP

#include <InstrumentComputePlaceId.hpp>
#include <InstrumentTaskExecution.hpp>
#include <InstrumentThreadId.hpp>

#include "hardware/places/ComputePlace.hpp"
#include "WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "system/ompss/SpawnFunction.hpp"
#include "tasks/Task.hpp"



class TaskFinalization {
public:
	static void disposeOrUnblockTask(Task *task, ComputePlace *computePlace);
	
};


#endif // TASK_FINALIZATION_HPP
