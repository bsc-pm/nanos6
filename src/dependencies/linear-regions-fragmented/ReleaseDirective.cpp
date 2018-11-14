/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef RELEASE_DIRECTIVE_HPP
#define RELEASE_DIRECTIVE_HPP


#include <nanos6/multidimensional-release.h>

#include "executors/threads/WorkerThread.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "tasks/Task.hpp"
#include "tasks/TaskImplementation.hpp"

#include "DataAccessRegistration.hpp"


template <DataAccessType ACCESS_TYPE, bool WEAK>
void release_access(void *base_address, __attribute__((unused)) long dim1size, long dim1start, long dim1end)
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	
	Task *task = currentWorkerThread->getTask();
	assert(task != nullptr);
	
	ComputePlace *computePlace = currentWorkerThread->getComputePlace();
	
	union {
		void *_asVoidPointer;
		char *_asCharPointer;
	} address;
	address._asVoidPointer = base_address;
	address._asCharPointer += dim1start;
	
	DataAccessRegion accessRegion(address._asVoidPointer, dim1end - dim1start);
	DataAccessRegistration::releaseAccessRegion(task, accessRegion, ACCESS_TYPE, WEAK, computePlace, computePlace->getDependencyData());
}


void nanos6_release_read_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<READ_ACCESS_TYPE, false>(base_address, dim1size, dim1start, dim1end);
}

void nanos6_release_write_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<WRITE_ACCESS_TYPE, false>(base_address, dim1size, dim1start, dim1end);
}

void nanos6_release_readwrite_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<READWRITE_ACCESS_TYPE, false>(base_address, dim1size, dim1start, dim1end);
}

void nanos6_release_concurrent_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<CONCURRENT_ACCESS_TYPE, false>(base_address, dim1size, dim1start, dim1end);
}

void nanos6_release_commutative_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<COMMUTATIVE_ACCESS_TYPE, false>(base_address, dim1size, dim1start, dim1end);
}


void nanos6_release_weak_read_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<READ_ACCESS_TYPE, true>(base_address, dim1size, dim1start, dim1end);
}

void nanos6_release_weak_write_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<WRITE_ACCESS_TYPE, true>(base_address, dim1size, dim1start, dim1end);
}

void nanos6_release_weak_readwrite_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<READWRITE_ACCESS_TYPE, true>(base_address, dim1size, dim1start, dim1end);
}

void nanos6_release_weak_commutative_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<COMMUTATIVE_ACCESS_TYPE, true>(base_address, dim1size, dim1start, dim1end);
}



#endif // RELEASE_DIRECTIVE_HPP
