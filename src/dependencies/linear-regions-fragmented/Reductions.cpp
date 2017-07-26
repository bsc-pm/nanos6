#include <nanos6.h>

#include <cassert>

#include "tasks/Task.hpp"
#include "executors/threads/WorkerThread.hpp"


void *nanos_get_original_reduction_address(const void *address)
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	assert(currentThread != nullptr);
	
	Task *currentTask = currentThread->getTask();
	assert(currentTask != nullptr);
	assert(currentTask->getThread() == currentThread);
	
	void *argsBlock = currentTask->getArgsBlock();
	
	if (argsBlock <= address && currentTask > address) {
		void *original = (void*)*(((void**)address) - 1);
		
		DataAccessRange range(original, 1);
		bool isReductionAccess = currentTask->getDataAccesses().
			_accesses.exists(range, [&](TaskDataAccesses::accesses_t::iterator position) -> bool {
					DataAccess *targetAccess = &(*position);
					return targetAccess->_type == REDUCTION_ACCESS_TYPE;
				});
		
		if (isReductionAccess) {
			return original;
		}
		else {
			return (void*)address;
		}
	}
	else {
		return (void*)address;
	}
}
