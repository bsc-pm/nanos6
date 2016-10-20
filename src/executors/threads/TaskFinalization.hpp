#ifndef TASK_FINALIZATION_HPP
#define TASK_FINALIZATION_HPP

#include "InstrumentTaskExecution.hpp"

#include "CPU.hpp"
#include "WorkerThread.hpp"
#include "scheduling/Scheduler.hpp"
#include "tasks/Task.hpp"



class TaskFinalization {
public:
	static void disposeOrUnblockTask(Task *task, CPU *cpu, WorkerThread *thread)
	{
		bool readyOrDisposable = true;
		
		// Follow up the chain of ancestors and dispose them as needed and wake up any in a taskwait that finishes in this moment
		while ((task != nullptr) && readyOrDisposable) {
			Task *parent = task->getParent();
			
			if (task->hasFinished()) {
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
			} else {
				// An ancestor in a taskwait that finishes at this point
				Scheduler::taskGetsUnblocked(task, cpu);
				readyOrDisposable = false;
			}
		}
	}
	
};


#endif // TASK_FINALIZATION_HPP
