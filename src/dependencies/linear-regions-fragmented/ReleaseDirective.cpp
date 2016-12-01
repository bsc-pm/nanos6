#ifndef RELEASE_DIRECTIVE_HPP
#define RELEASE_DIRECTIVE_HPP


#include <nanos6/multidimensional-release.h>

#include "tasks/Task.hpp"
#include "DataAccessRegistration.hpp"


template <DataAccessType ACCESS_TYPE, bool WEAK>
void release_access(void *base_address, __attribute__((unused)) long dim1size, long dim1start, long dim1end)
{
	WorkerThread *currentWorkerThread = WorkerThread::getCurrentWorkerThread();
	assert(currentWorkerThread != nullptr);
	Task *task = currentWorkerThread->getTask();
	
	union {
		void *_asVoidPointer;
		char *_asCharPointer;
	} address;
	address._asVoidPointer = base_address;
	address._asCharPointer += dim1start;
	
	DataAccessRange accessRange(address._asVoidPointer, dim1end - dim1start);
	DataAccessRegistration::releaseAccessRange(task, accessRange, ACCESS_TYPE, WEAK);
}


void nanos_release_read_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<READ_ACCESS_TYPE, false>(base_address, dim1size, dim1start, dim1end);
}

void nanos_release_write_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<WRITE_ACCESS_TYPE, false>(base_address, dim1size, dim1start, dim1end);
}

void nanos_release_readwrite_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<READWRITE_ACCESS_TYPE, false>(base_address, dim1size, dim1start, dim1end);
}


void nanos_release_weak_read_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<READ_ACCESS_TYPE, true>(base_address, dim1size, dim1start, dim1end);
}

void nanos_release_weak_write_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<WRITE_ACCESS_TYPE, true>(base_address, dim1size, dim1start, dim1end);
}

void nanos_release_weak_readwrite_1(void *base_address, long dim1size, long dim1start, long dim1end)
{
	release_access<READWRITE_ACCESS_TYPE, true>(base_address, dim1size, dim1start, dim1end);
}





#endif // RELEASE_DIRECTIVE_HPP
