#include <cassert>

#include <InstrumentDependenciesByAccess.hpp>

#include "api/nanos6_rt_interface.h"
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"


template <DataAccess::type_t ACCESS_TYPE>
void register_access(void *handler, void *start, size_t length)
{
	assert(handler != 0);
	Task *task = (Task *) handler;
	
	Instrument::registerTaskAccess(task->getInstrumentationTaskId(), ACCESS_TYPE, start, length);
	
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != 0); // NOTE: The "main" task is not created by a WorkerThread, but in any case it is not supposed to have dependencies
	
	WorkerThread::dependency_domain_t &dependencyDomain = currentWorkerThread->getDependencyDomain();
	DataAccessSequence &accessSequence = dependencyDomain[start];
	
	DataAccess *dataAccess;
	bool satisfied = accessSequence.addTaskAccess(task, ACCESS_TYPE, /* OUT */ dataAccess);
	if (dataAccess != 0) {
		// A new data access, as opposed to a repeated or upgraded one
		task->addDataAccess(dataAccess);
	}
	
	if (!satisfied) {
		task->increasePredecessors();
	}
}


void nanos_register_read_depinfo(void *handler, void *start, size_t length)
{
	register_access<DataAccess::READ>(handler, start, length);
}


void nanos_register_write_depinfo(void *handler, void *start, size_t length)
{
	register_access<DataAccess::WRITE>(handler, start, length);
}


void nanos_register_readwrite_depinfo(void *handler, void *start, size_t length)
{
	register_access<DataAccess::READWRITE>(handler, start, length);
}


