/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <stdlib.h>

#include <nanos6.h>

#include "DataAccessRegistration.hpp"
#include "Dependencies.hpp"
#include "ReductionSpecific.hpp"
#include "dependencies/DataAccessType.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include <InstrumentDependenciesByAccess.hpp>

template <DataAccessType ACCESS_TYPE, bool WEAK>
void register_access(void *handler, void *start, size_t length, int symbolIndex,
	reduction_type_and_operator_index_t reductionTypeAndOperatorIndex = no_reduction_type_and_operator,
	reduction_index_t reductionIndex = no_reduction_index)
{
	assert(handler != 0);
	Task *task = (Task *)handler;

	if (start == nullptr) {
		return;
	}

	if (length == 0) {
		return;
	}

	bool weak = (WEAK && !task->isFinal()) || task->isTaskloopSource();
	Instrument::registerTaskAccess(task->getInstrumentationTaskId(), ACCESS_TYPE, weak, start, length);

	DataAccessRegistration::registerTaskDataAccess(task, ACCESS_TYPE, weak, start, length, reductionTypeAndOperatorIndex, reductionIndex, symbolIndex);
}

void nanos6_register_read_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<READ_ACCESS_TYPE, false>(handler, start, length, symbolIndex);
}

void nanos6_register_write_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<WRITE_ACCESS_TYPE, false>(handler, start, length, symbolIndex);
}

void nanos6_register_readwrite_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<READWRITE_ACCESS_TYPE, false>(handler, start, length, symbolIndex);
}

void nanos6_register_weak_read_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<READ_ACCESS_TYPE, true>(handler, start, length, symbolIndex);
}

void nanos6_register_weak_write_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<WRITE_ACCESS_TYPE, true>(handler, start, length, symbolIndex);
}

void nanos6_register_weak_readwrite_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<READWRITE_ACCESS_TYPE, true>(handler, start, length, symbolIndex);
}

void nanos6_register_concurrent_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<CONCURRENT_ACCESS_TYPE, false>(handler, start, length, symbolIndex);
}

void nanos6_register_commutative_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<COMMUTATIVE_ACCESS_TYPE, false>(handler, start, length, symbolIndex);
}

void nanos6_register_weak_commutative_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<COMMUTATIVE_ACCESS_TYPE, true>(handler, start, length, symbolIndex);
}

void nanos6_register_region_reduction_depinfo1(
	int reduction_operation,
	int reduction_index,
	void *handler,
	int symbol_index,
	__attribute__((unused)) char const *region_text,
	void *base_address,
	long dim1size,
	__attribute__((unused)) long dim1start,
	__attribute__((unused)) long dim1end)
{
	// Currently we only support contiguous regions without offset
	assert(dim1start == 0L);

	register_access<REDUCTION_ACCESS_TYPE, false>(handler, base_address, dim1size, symbol_index, reduction_operation, reduction_index);
}

void nanos6_register_region_weak_reduction_depinfo1(
	int reduction_operation,
	int reduction_index,
	void *handler,
	int symbol_index,
	__attribute__((unused)) char const *region_text,
	void *base_address,
	long dim1size,
	__attribute__((unused)) long dim1start,
	__attribute__((unused)) long dim1end)
{
	assert(dim1start == 0L);
	register_access<REDUCTION_ACCESS_TYPE, true>(handler, base_address, dim1size, symbol_index, reduction_operation, reduction_index);
}
