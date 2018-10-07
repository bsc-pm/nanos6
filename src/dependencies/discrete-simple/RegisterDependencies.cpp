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

#include <Dependencies.hpp>

template <DataAccessType ACCESS_TYPE, bool WEAK>
void register_access(void *handler, void *start, __attribute__((unused)) size_t length, __attribute__((unused)) int symbolIndex)
{
	assert(handler != 0);
	Task *task = (Task *) handler;
	
	if (start == nullptr) {
		return;
	}
	
	DataAccessRegistration::registerTaskDataAccess(task, ACCESS_TYPE, start);
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
	register_access<READWRITE_ACCESS_TYPE, false>(handler, start, length, symbolIndex);
}

void nanos6_register_commutative_depinfo(void *handler, void *start, size_t length, int symbolIndex)
{
	register_access<READWRITE_ACCESS_TYPE, false>(handler, start, length, symbolIndex);
}

void nanos6_register_region_reduction_depinfo1(
		__attribute__((unused)) int reduction_operation,
		__attribute__((unused)) int reduction_index,
		__attribute__((unused)) void *handler,
		__attribute__((unused)) int symbol_index,
		__attribute__((unused)) char const *region_text,
		__attribute__((unused)) void *base_address,
		__attribute__((unused)) long dim1size,
		__attribute__((unused)) long dim1start,
		__attribute__((unused)) long dim1end
) {
	assert(0);
}

void nanos6_register_region_weak_reduction_depinfo1(
		__attribute__((unused)) int reduction_operation,
		__attribute__((unused)) int reduction_index,
		__attribute__((unused)) void *handler,
		__attribute__((unused)) int symbol_index,
		__attribute__((unused)) char const *region_text,
		__attribute__((unused)) void *base_address,
		__attribute__((unused)) long dim1size,
		__attribute__((unused)) long dim1start,
		__attribute__((unused)) long dim1end
) {
	assert(0);
}
