#ifndef TASK_FINALIZATION_IMPLEMENTATION_HPP
#define TASK_FINALIZATION_IMPLEMENTATION_HPP

#include "DataAccessRegistration.hpp"
#include "TaskFinalization.hpp"


void TaskFinalization::disposeOrUnblockTask(Task *task, CPU *cpu, WorkerThread *thread)
{
	assert(task != nullptr);
	assert(cpu != nullptr);
	assert(thread != nullptr);
	
	bool readyOrDisposable = true;
	
	// Follow up the chain of ancestors and dispose them as needed and wake up any in a taskwait that finishes in this moment
	while ((task != nullptr) && readyOrDisposable) {
		Task *parent = task->getParent();
		
		if (task->hasFinished()) {
			// NOTE: Handle task removal before unlinking from parent
			DataAccessRegistration::handleTaskRemoval(task);
			
			readyOrDisposable = task->unlinkFromParent();
			
			Instrument::destroyTask(
				task->getInstrumentationTaskId(),
				(cpu != nullptr ? cpu->_virtualCPUId : ~0UL),
				(thread != nullptr ? thread->getInstrumentationId() : Instrument::thread_id_t())
			);
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
			Scheduler::taskGetsUnblocked(task, cpu);
			readyOrDisposable = false;
		}
	}
}


#endif // TASK_FINALIZATION_IMPLEMENTATION_HPP
