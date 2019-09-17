/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <nanos6.h>
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include "../DataAccessType.hpp"
#include "DataAccessRegistration.hpp"
#include "ReductionSpecific.hpp"

#include <Dependencies.hpp>

#include <stdlib.h>
#include <iostream>

template <DataAccessType ACCESS_TYPE, bool WEAK>
void register_access(void *handler, void *start, size_t length, __attribute__((unused)) int symbolIndex, 
					 reduction_type_and_operator_index_t reductionTypeAndOperatorIndex = no_reduction_type_and_operator,
					 reduction_index_t reductionIndex = no_reduction_index)
{
	assert(handler != 0);
	Task *task = (Task *) handler;
	
	if (start == nullptr) {
		return;
	}

	if(length == 0) {
		return;
	}
	
	DataAccessRegistration::registerTaskDataAccess(task, ACCESS_TYPE, WEAK && !task->isFinal() && !task->isTaskfor(), start, length, reductionTypeAndOperatorIndex, reductionIndex);
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
	register_access<READ_ACCESS_TYPE, false>(handler, start, length, symbolIndex);
}

void nanos6_register_weak_write_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<WRITE_ACCESS_TYPE, false>(handler, start, length, symbolIndex);
}

void nanos6_register_weak_readwrite_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<READWRITE_ACCESS_TYPE, false>(handler, start, length, symbolIndex);
}

void nanos6_register_concurrent_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<READWRITE_ACCESS_TYPE, false>(handler, start, length, symbolIndex);
}

void nanos6_register_commutative_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<READWRITE_ACCESS_TYPE, false>(handler, start, length, symbolIndex);
}

void nanos6_register_region_reduction_depinfo1(
		int reduction_operation,
		int reduction_index,
		void *handler,
		int symbol_index,
		__attribute__((unused)) char const *region_text,
		__attribute__((unused)) void *base_address,
		long dim1size,
		__attribute__((unused)) long dim1start,
		__attribute__((unused)) long dim1end
) {
	// Currently we only support contiguous regions without offset
	assert(dim1start == 0L);

	register_access<REDUCTION_ACCESS_TYPE, false>(handler, base_address, dim1size, symbol_index, reduction_operation, reduction_index);
}

void nanos6_register_region_weak_reduction_depinfo1(
		__attribute__((unused)) int reduction_operation,
		__attribute__((unused)) int reduction_index,
		__attribute__((unused)) void *handler,
		__attribute__((unused)) int symbol_index,
		__attribute__((unused)) char const *region_text,
		__attribute__((unused)) void *base_address,
		__attribute__((unused)) long dim1size,
		long dim1start,
		__attribute__((unused)) long dim1end
) {
	assert(dim1start == 0L);
	
	// We don't support weak reductions, but we cannot safely ignore them.
	// So we need to die.

	std::cerr << "The selected dependency implementation has no support for nested-reduction (aka. weakreduction)." << std::endl;
	exit(1);
}
