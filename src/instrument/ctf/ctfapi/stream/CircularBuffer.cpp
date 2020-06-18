/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif


#include <sys/mman.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "CircularBuffer.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#define PAGE_SHIFT (12)
#define PAGE_SIZE (1 << PAGE_SHIFT)
#define NSUBBUF_SHIFT (2)
#define NSUBBUF (1 << NSUBBUF_SHIFT)
#define ALIGN_SHIFT (PAGE_SHIFT + NSUBBUF_SHIFT)
#define ALIGN_SIZE (1 << ALIGN_SHIFT)

void CircularBuffer::initializeFile(const char *path)
{
	int tmpfd;

	// open backing file for buffer
	tmpfd = open(path, O_CREAT | O_WRONLY, 0660);
	FatalErrorHandler::failIf(tmpfd == -1,
		" circular buffer: when opening buffer file: ",
		strerror(errno)
	);

	// set initial values
	fd = tmpfd;
	fileOffset = 0;
}

void CircularBuffer::initializeBuffer(uint64_t size)
{
	void *tmp;
	uint64_t sizeAligned;

	if (size < ALIGN_SIZE) {
		std::cerr << "WARNING: supplied circular buffer size is smaller than minimum. Using default of " << ALIGN_SIZE/1024 << " KiB" << std::endl;
		size = ALIGN_SIZE;
	}

	// we want the buffer to be multiple of both the PAGE_SIZE and the
	// number of subbuffers. This is because we will divide the buffer into
	// a number of subbufers and each subbufer should be multiple of the
	// page size for performance of calculating modulus

	size_t nAlign = (size + (ALIGN_SIZE - 1)) >> ALIGN_SHIFT;
	sizeAligned = nAlign * ALIGN_SIZE;

	// TODO allocate memory on each CPU (madvise or specific
	// instrument function?)

	tmp = mmap(NULL, sizeAligned, PROT_READ | PROT_WRITE,
		   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	FatalErrorHandler::failIf(
		buffer == MAP_FAILED,
		" circular buffer: when allocating tracing buffer: ",
		strerror(errno)
	);

	// set initial values
	mask          = sizeAligned - 1;
	buffer        = (char *) tmp;
	bufferSize    = sizeAligned;
	subBufferSize = sizeAligned/NSUBBUF;
	subBufferMask = subBufferSize - 1;

	resetPointers();
}

void CircularBuffer::initialize(uint64_t size, const char *path)
{
	initializeBuffer(size);
	initializeFile(path);
}

void CircularBuffer::shutdown()
{
	int ret;

	flushAll();
	ret = close(fd);
	FatalErrorHandler::warnIf(
		ret == -1,
		ret, " circular buffer: when closing backing file: ",
		strerror(errno)
	);
	ret = munmap(buffer, bufferSize);
	FatalErrorHandler::warnIf(
		ret == -1,
		ret, " circular buffer: when unmapping circular buffer: ",
		strerror(errno)
	);
	bufferSize = 0;
}

void CircularBuffer::resetPointers()
{
	head = 0;
	tail = 0;
	hole = 0;
	wall = bufferSize;
}

void CircularBuffer::flushToFile(char *buf, size_t size)
{
	off_t offset = 0;
	size_t rem = size;
	ssize_t ret;

	assert(bufferSize != 0);

	if (!size)
		return;

	do {
		ret = pwrite(fd, (const char *)buf + offset, rem, fileOffset);
		FatalErrorHandler::failIf(ret < 0,
			"circular buffer: when writing to file: ", strerror(errno)
		);

		offset     += ret;
		rem        -= ret;
		fileOffset += ret;
	} while (rem > 0);
}

bool CircularBuffer::checkIfNeedsFlush()
{
	uint64_t size;

	// if wraps it needs to flush
	if (wraps())
		return true;

	// otherwise let's check if the wirtten size exceeds the subbuffer size
	size = ((head - tail) & ~subBufferMask);
	return (size > 0);
}

bool CircularBuffer::alloc(uint64_t size)
{
	uint64_t next_wall;

	assert(size <= bufferSize);

	// there is enough space in the buffer?
	if (head + size - tail > bufferSize) {
		return false;
	}

	// is this space contiguous?
	next_wall = (head & ~mask) + bufferSize;
	if (next_wall - head < size) {
		hole = head;
		head = next_wall;
		// if not, is the next space contiguous?
		if (head + size - tail > bufferSize) {
			return false;
		}
	}

	// yes, we have space!
	return true;
}

void CircularBuffer::flushUpToTheWrap()
{
	uint64_t seg;

	if (!wraps())
		return;

	seg = (tail < hole) ? hole : wall;
	flushToFile(buffer + (tail & mask), seg - tail);
	tail = wall;
	wall += bufferSize;
}

void CircularBuffer::flushAll()
{
	// if the buffer wraps flush up to the wall or hole first
	flushUpToTheWrap();

	// next flush up to head
	flushToFile(buffer + (tail & mask), head - tail);

	// move pointers to the beginning of the buffer; we want head to be as
	// far as possible from the wall
	resetPointers();
}

void CircularBuffer::flushFilledSubBuffers()
{
	uint64_t size;

	// if the buffer wraps flush up to the wall or hole first
	flushUpToTheWrap();

	// next, flush up to the next subbuffer. Here we priorize flushing
	// size aligned blocks rather than flushing everything
	size = ((head - tail) & ~subBufferMask);
	flushToFile(buffer + (tail & mask), size);
	tail += size;

	// if we have flushed everything, return pointers to the beginning of
	// the buffer, we want head to be as far as possible from the wall.
	if (tail == head) {
		resetPointers();
	}

	// note that size will always advance in multiples of subBufferSize,
	// hence, all flushes will be aligned but for flushes with holes (whose
	// start address will be aligned but not its size)
}
