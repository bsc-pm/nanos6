#include <cassert>

#include <InstrumentDependenciesByAccess.hpp>

#include "api/nanos6_rt_interface.h"
#include "dependencies/DataAccessType.hpp"
#include "dependencies/DataAccessRegistration.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"


template <DataAccessType ACCESS_TYPE>
void register_access(void *handler, void *start, size_t length)
{
	assert(handler != 0);
	Task *task = (Task *) handler;
	
	Instrument::registerTaskAccess(task->getInstrumentationTaskId(), ACCESS_TYPE, start, length);
	
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != 0); // NOTE: The "main" task is not created by a WorkerThread, but in any case it is not supposed to have dependencies
	
	DataAccessRange accessRange(start, length);
	
	DataAccessSequence *accessSequence = 0;
	Task *parent = task->getParent();
	if (parent != 0) {
		for (DataAccess &parentAccess : parent->getDataAccesses()) {
			DataAccessSequence *parentSequence = parentAccess._dataAccessSequence;
			assert(parentSequence != 0);
			
			if (parentSequence->_accessRange == accessRange) {
				accessSequence = &parentAccess._subaccesses;
				break;
			}
		}
	}
	
	if (accessSequence == 0) {
		// An access that is not a subset of the parent accesses, therefore
		// (if the code is correct) it must be temporary data created by the parent
		WorkerThread::dependency_domain_t &dependencyDomain = currentWorkerThread->getDependencyDomain();
		accessSequence = &dependencyDomain[accessRange];
	}
	
	DataAccess *dataAccess;
	bool satisfied = DataAccessRegistration::registerTaskDataAccess(task, ACCESS_TYPE, accessSequence, /* OUT */ dataAccess);
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
	register_access<READ_ACCESS_TYPE>(handler, start, length);
}


void nanos_register_write_depinfo(void *handler, void *start, size_t length)
{
	register_access<WRITE_ACCESS_TYPE>(handler, start, length);
}


void nanos_register_readwrite_depinfo(void *handler, void *start, size_t length)
{
	register_access<READWRITE_ACCESS_TYPE>(handler, start, length);
}


