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

#include "lowlevel/FatalErrorHandler.hpp"

#include "CTFStream.hpp"

#define PAGE_SHIFT (12)
#define PAGE_SIZE (1 << PAGE_SHIFT)

#define NSUBBUF_SHIFT (2)
#define NSUBBUF (1 << NSUBBUF_SHIFT)
#define ALIGN_SHIFT (PAGE_SHIFT + NSUBBUF_SHIFT)
#define ALIGN_SIZE (1 << ALIGN_SHIFT)



void CTFAPI::CTFStream::initialize(size_t size, ctf_cpu_id_t cpu)
{
	int fd;
	void *mrbPart;
	size_t sizeAligned;

	//size_t nPages = (size + (PAGE_SIZE - 1)) >> PAGE_SHIFT;
	//sizeAligned = nPages * PAGE_SIZE;

	if (size < ALIGN_SIZE) {
		std::cerr << "WARNING: supplied CTF buffer size is smaller than minimum. Using default of " << ALIGN_SIZE/1024 << " KiB" << std::endl;
		size = ALIGN_SIZE;
	}

	// we want the buffer to be multiple of both the PAGE_SIZE and the
	// number of subbuffers. This is because we will divide the buffer into
	// a number of subbufers and each subbufer should be multiple of the
	// page size for performance of calculating modulus
	size_t nAlign = (size + (ALIGN_SIZE - 1)) >> ALIGN_SHIFT;
	sizeAligned = nAlign * ALIGN_SIZE;
	mrbSize = sizeAligned * 2;

	// allocate backing physical memory for the event buffer
	fd = open("/tmp", O_TMPFILE | O_RDWR | O_EXCL, 0600);
	if (fd == -1) {
		FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: open: ") + strerror(errno));
	}
	if (ftruncate(fd, sizeAligned)) {
		FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: ftruncate: ") + strerror(errno));
	}

	// allocate virtual addresses for the ring buffer (2x requestes buffer size)
	mrb = mmap(NULL, mrbSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (mrb == MAP_FAILED) {
		FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: mmap base: ") + strerror(errno));
	}

	// mmap physical pages to virtual addresses
	mrbPart = mmap(mrb, sizeAligned, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
	if (mrbPart != mrb) {
		FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: mmap part 1: ") + strerror(errno));
	}
	mrbPart = mmap(((char *) mrb) + sizeAligned, sizeAligned, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
	if (mrbPart != ((char *) mrb) + sizeAligned) {
		FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: mmap part 2: ") + strerror(errno));
	}

	close(fd);

	// set initial values
	head = 0;
	tail = 0;
	mask = sizeAligned - 1;
	buffer = (char *) mrb;
	bufferSize = sizeAligned;
	subBufferSize = sizeAligned/NSUBBUF;
	subBufferMask = subBufferSize - 1;
	fileOffset = 0;
	cpuId = cpu;
}

void CTFAPI::CTFStream::shutdown(void)
{
	if (!bufferSize)
		return;

	munmap(mrb, mrbSize);
	head = 0;
	tail = 0;
	bufferSize = 0;
}

void CTFAPI::CTFStream::doWrite(int fd, const char *buf, size_t size)
{
	off_t offset = 0;
	size_t rem = size;
	ssize_t ret;

	assert(bufferSize != 0);

	do {
		ret = pwrite(fd, (const char *)buf + offset, rem, fileOffset);
		if (ret < 0) {
			FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: flush buffer: ") + strerror(errno));
		}

		offset     += ret;
		rem        -= ret;
		fileOffset += ret;
	} while (rem > 0);
}

void CTFAPI::CTFStream::flushAll()
{
	size_t size;

	assert(bufferSize != 0);

	size = head - tail;
	doWrite(fdOutput, buffer + (tail & mask), size);
	tail = head;
}

void CTFAPI::CTFStream::flushFilledSubBuffers()
{
	size_t size;

	assert(bufferSize != 0);

	size = head - tail;
	size &= ~subBufferMask;
	assert(size > 0);
	doWrite(fdOutput, buffer + (tail & mask), size);
	tail += size;
}

bool CTFAPI::CTFStream::checkIfNeedsFlush()
{
	return (head - tail >= subBufferSize);
}

bool CTFAPI::CTFStream::checkFreeSpace(size_t size)
{
	assert(bufferSize != 0);

	return ((head + size - tail) <= bufferSize);
}
