#include <cassert>

#include <InstrumentDependenciesByAccess.hpp>

#include "api/nanos6_rt_interface.h"
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"

#include "../DataAccessType.hpp"
#include "DataAccessRegistration.hpp"


template <DataAccessType ACCESS_TYPE, bool WEAK>
void register_access(void *handler, void *start, size_t length)
{
	assert(handler != 0);
	Task *task = (Task *) handler;
	
	Instrument::registerTaskAccess(task->getInstrumentationTaskId(), ACCESS_TYPE, WEAK, start, length);
	
	if (start == nullptr) {
		return;
	}
	if (length == 0) {
		return;
	}
	
	DataAccessRange accessRange(start, length);
	DataAccessRegistration::registerTaskDataAccess(task, ACCESS_TYPE, WEAK, accessRange);
    task->addDataSize(length);
}


void nanos_register_read_depinfo(void *handler, void *start, size_t length)
{
	register_access<READ_ACCESS_TYPE, false>(handler, start, length);
}


void nanos_register_write_depinfo(void *handler, void *start, size_t length)
{
	register_access<WRITE_ACCESS_TYPE, false>(handler, start, length);
}


void nanos_register_readwrite_depinfo(void *handler, void *start, size_t length)
{
	register_access<READWRITE_ACCESS_TYPE, false>(handler, start, length);
}


void nanos_register_weak_read_depinfo(void *handler, void *start, size_t length)
{
	register_access<READ_ACCESS_TYPE, true>(handler, start, length);
}


void nanos_register_weak_write_depinfo(void *handler, void *start, size_t length)
{
	register_access<WRITE_ACCESS_TYPE, true>(handler, start, length);
}


void nanos_register_weak_readwrite_depinfo(void *handler, void *start, size_t length)
{
	register_access<READWRITE_ACCESS_TYPE, true>(handler, start, length);
}


