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
	static void disposeOrUnblockTask(Task *task, ComputePlace *computePlace)
	{
		bool readyOrDisposable = true;
		
		// Follow up the chain of ancestors and dispose them as needed and wake up any in a taskwait that finishes in this moment
		while ((task != nullptr) && readyOrDisposable) {
			Task *parent = task->getParent();
			
			if (task->hasFinished()) {
				readyOrDisposable = task->unlinkFromParent();
				Instrument::destroyTask(task->getInstrumentationTaskId());
				// NOTE: The memory layout is defined in nanos_create_task
				task->~Task();
				free(task->getArgsBlock()); // FIXME: Need a proper object recycling mechanism here
				task = parent;
				
				// A task without parent must be a spawned function
				if (parent == nullptr) {
					SpawnedFunctions::_pendingSpawnedFuncions--;
				}
			} else {
				// An ancestor in a taskwait that finishes at this point
				Scheduler::taskGetsUnblocked(task, computePlace);
				readyOrDisposable = false;
			}
		}
	}
	
};


#endif // TASK_FINALIZATION_HPP
