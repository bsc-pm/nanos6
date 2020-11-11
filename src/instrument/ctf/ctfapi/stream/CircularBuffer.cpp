/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif


#include <algorithm>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <numa.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
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
	_fd = tmpfd;
	_fileOffset = 0;
}

static void prefaultMemory(char *addr, size_t size)
{
	for (size_t i = 0; i < size; i+= PAGE_SIZE) {
		addr[i] = 1;
	}
}

void CircularBuffer::initializeBuffer(uint64_t size, int node)
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

	_node = node;
	// Circular buffers of threads not bound to any core (such as external
	// threads) are given a NUMA node id of -1 to specify that its memory
	// should not be allocated in any specific NUMA node
	if (node != -1) {
		// Allocate memory on a specific numa node
		tmp = numa_alloc_onnode(sizeAligned, node);
		if (tmp == nullptr) {
			FatalErrorHandler::fail(
				" circular buffer: when allocating tracing buffer: ",
				strerror(errno)
			);
		}
		prefaultMemory((char *) tmp, sizeAligned);
	} else {
		// Allocate memory without binding, let it be allocated as it is accessed
		tmp = mmap(NULL, sizeAligned, PROT_READ | PROT_WRITE,
			   MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
		if (tmp == MAP_FAILED) {
			FatalErrorHandler::fail(
				" circular buffer: when allocating tracing buffer: ",
				strerror(errno)
			);
		}
	}

	// Set initial values
	_mask          = sizeAligned - 1;
	_buffer        = (char *) tmp;
	_bufferSize    = sizeAligned;
	_subBufferSize = sizeAligned/NSUBBUF;
	_subBufferMask = _subBufferSize - 1;

	resetPointers();
}

void CircularBuffer::initialize(uint64_t size, int node, const char *path)
{
	initializeBuffer(size, node);
	initializeFile(path);
}

void CircularBuffer::shutdown()
{
	int ret;

	flushAll();
	ret = close(_fd);
	FatalErrorHandler::warnIf(
		ret == -1,
		ret, " circular buffer: when closing backing file: ",
		strerror(errno)
	);
	if (_node != -1) {
		numa_free(_buffer, _bufferSize);
	} else {
		ret = munmap(_buffer, _bufferSize);
		if (ret == -1) {
			FatalErrorHandler::warn(
				" circular buffer: when unmapping circular buffer: ",
				strerror(errno)
			);
		}
	}
	_bufferSize = 0;
}

void CircularBuffer::resetPointers()
{
	_head = 0;
	_tail = 0;
	_hole = 0;
	_wall = _bufferSize;
}

void CircularBuffer::flushToFile(char *buf, size_t size)
{
	off_t offset = 0;
	size_t rem = size;
	ssize_t ret;

	assert(_bufferSize != 0);

	if (!size)
		return;

	do {
		ret = pwrite(_fd, (const char *)buf + offset, rem, _fileOffset);
		FatalErrorHandler::failIf(ret < 0,
			"circular buffer: when writing to file: ", strerror(errno)
		);

		offset      += ret;
		rem         -= ret;
		_fileOffset += ret;
	} while (rem > 0);
}

bool CircularBuffer::checkIfNeedsFlush()
{
	uint64_t size;

	// If wraps it needs to flush
	if (wraps())
		return true;

	// Otherwise let's check if the wirtten size exceeds the subbuffer size
	size = ((_head - _tail) & ~_subBufferMask);
	return (size > 0);
}

bool CircularBuffer::alloc(uint64_t size)
{
	uint64_t next_wall;

	assert(size <= _bufferSize);

	// There is enough space in the buffer?
	if (_head + size - _tail > _bufferSize) {
		return false;
	}

	// Is this space contiguous?
	next_wall = (_head & ~_mask) + _bufferSize;
	if (next_wall - _head < size) {
		_hole = _head;
		_head = next_wall;
		// if not, is the next space contiguous?
		if (_head + size - _tail > _bufferSize) {
			return false;
		}
	}

	// Yes, we have space!
	return true;
}

uint64_t CircularBuffer::allocAtLeast(uint64_t minSize)
{
	uint64_t nextWall;
	uint64_t available;

	assert(minSize <= _bufferSize);

	// Is there enough space in the buffer?
	if (_head + minSize - _tail > _bufferSize) {
		return 0;
	}

	// Is there enough contiguous space to service the requested minimum size?
	nextWall = (_head & ~_mask) + _bufferSize;
	if (nextWall - _head < minSize) {
		// If not, mark this segment as a hole and move forward
		_hole = _head;
		_head = nextWall;
		// We cannot cross the border again so what's left is what we have
		available = _bufferSize - (_head - _tail);
		// Check again the available size, after moving the _head there might
		// no longer be enough space
		available = (available >= minSize)? available : 0;
	} else {
		// If yes, get the minimum between the real space left and and
		// the maximum contiguous space
		available = std::min(_bufferSize - (_head - _tail), nextWall - _head);
	}

	return available;
}

void CircularBuffer::flushUpToTheWrap()
{
	uint64_t seg;

	if (!wraps())
		return;

	seg = (_tail < _hole) ? _hole : _wall;
	flushToFile(_buffer + (_tail & _mask), seg - _tail);
	_tail = _wall;
	_wall += _bufferSize;
}

void CircularBuffer::flushAll()
{
	// If the buffer wraps flush up to the wall or hole first
	flushUpToTheWrap();

	// Next flush up to _head
	flushToFile(_buffer + (_tail & _mask), _head - _tail);

	// Move pointers to the beginning of the buffer; we want head to be as
	// far as possible from the wall
	resetPointers();
}

void CircularBuffer::flushFilledSubBuffers()
{
	uint64_t size;

	// If the buffer wraps flush up to the wall or _hole first
	flushUpToTheWrap();

	// Next, flush up to the next subbuffer. Here we priorize flushing
	// size aligned blocks rather than flushing everything
	size = ((_head - _tail) & ~_subBufferMask);
	flushToFile(_buffer + (_tail & _mask), size);
	_tail += size;

	// If we have flushed everything, return pointers to the beginning of
	// the buffer, we want head to be as far as possible from the wall
	if (_tail == _head) {
		resetPointers();
	}

	// Note that size will always advance in multiples of subBufferSize,
	// hence, all flushes will be aligned but for flushes with holes (whose
	// start address will be aligned but not its size)
}
