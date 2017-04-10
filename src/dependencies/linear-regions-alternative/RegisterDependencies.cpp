#include <cassert>

#include <InstrumentDependenciesByAccess.hpp>

#include <nanos6.h>
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"

#include "../DataAccessType.hpp"
#include "DataAccessRegistration.hpp"


template <DataAccessType ACCESS_TYPE, bool WEAK, typename... ReductionInfo>
void register_access(void *handler, void *start, size_t length, ReductionInfo... reductionInfo)
{
	assert(handler != 0);
	Task *task = (Task *) handler;
	
	Instrument::registerTaskAccess(task->getInstrumentationTaskId(), ACCESS_TYPE, WEAK && !task->isFinal(), start, length);
	
	if (start == nullptr) {
		return;
	}
	if (length == 0) {
		return;
	}
	
	DataAccessRange accessRange(start, length);
	DataAccessRegistration::registerTaskDataAccess(task, ACCESS_TYPE, WEAK && !task->isFinal(), accessRange, reductionInfo...);
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


void nanos_register_concurrent_depinfo(void *handler, void *start, size_t length)
{
	register_access<CONCURRENT_ACCESS_TYPE, false>(handler, start, length);
}

void nanos_register_region_reduction_depinfo1(
		int reduction_operation, int reduction_index,
		void *handler,
		int symbol_index,
		char const *region_text,
		void *base_address,
		long dim1size, long dim1start, long dim1end)
{
	FatalErrorHandler::failIf(dim1size > sizeof(long long), "Array reductions not supported");

	register_access<REDUCTION_ACCESS_TYPE, false>(handler, base_address, dim1size, reduction_operation);
}
