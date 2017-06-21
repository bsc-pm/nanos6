#include "Taskloop.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"

#include <DataAccessRegistration.hpp>

void Taskloop::unregisterDataAccesses()
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	
	ComputePlace *computePlace = currentWorkerThread->getComputePlace();
	assert(computePlace != nullptr);
	
	DataAccessRegistration::unregisterTaskDataAccesses(this, computePlace);
}

