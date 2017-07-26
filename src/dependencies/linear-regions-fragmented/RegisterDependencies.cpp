/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <InstrumentDependenciesByAccess.hpp>

#include <nanos6.h>
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"

#include "../DataAccessType.hpp"
#include "DataAccessRegistration.hpp"
#include "ReductionSpecific.hpp"


template <DataAccessType ACCESS_TYPE, bool WEAK>
void register_access(void *handler, void *start, size_t length, int reductionTypeAndOperatorIndex = no_reduction_type_and_operator)
{
	assert(handler != 0);
	Task *task = (Task *) handler;
	
	if (WEAK && task->isTaskloop()) {
		std::cerr << "Warning: task loop cannot have weak dependencies. Changing them to strong dependencies." << std::endl;
	}
	
	Instrument::registerTaskAccess(task->getInstrumentationTaskId(), ACCESS_TYPE, WEAK && !task->isFinal() && !task->isTaskloop(), start, length);
	
	if (start == nullptr) {
		return;
	}
	if (length == 0) {
		return;
	}
	
	DataAccessRange accessRange(start, length);
	DataAccessRegistration::registerTaskDataAccess(task, ACCESS_TYPE, WEAK && !task->isFinal() && !task->isTaskloop(), accessRange, reductionTypeAndOperatorIndex);
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
		int reduction_operation,
		__attribute__((unused)) int reduction_index,
		void *handler,
		__attribute__((unused)) int symbol_index,
		__attribute__((unused)) char const *region_text,
		void *base_address,
		long dim1size,
		__attribute__((unused)) long dim1start,
		__attribute__((unused)) long dim1end
) {
	// Currently we only support non-arrays
	assert(dim1start == 0L);
	
	register_access<REDUCTION_ACCESS_TYPE, false>(handler, base_address, dim1size, reduction_operation);
}
