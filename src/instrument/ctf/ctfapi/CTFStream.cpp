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



void CTFAPI::CTFStream::initialize(size_t size, ctf_cpu_id_t cpu)
{
	int fd;
	size_t nPages;
	void *mrbPart;
	size_t sizeAligned;
	const size_t thresholdDefault = 1024;

	nPages = (size + (PAGE_SIZE - 1)) >> PAGE_SHIFT;
	sizeAligned = nPages * PAGE_SIZE;
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
	lost = 0;
	head = 0;
	tail = 0;
	// TODO read threshold from env
	threshold = (thresholdDefault > sizeAligned)? sizeAligned : thresholdDefault;
	tailCommited = 0;
	mask = sizeAligned - 1;
	buffer = (char *) mrb;
	bufferSize = sizeAligned;
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
	tailCommited = 0;
	bufferSize = 0;
}

bool CTFAPI::CTFStream::checkFreeSpace(size_t size)
{
	assert(bufferSize != 0);

	if ((head + size - tailCommited) >= threshold)
		flushData();

	if ((head + size - tail) >= bufferSize) {
		lost++;
		return false;
	}

	return true;
}

void CTFAPI::CTFStream::doWrite(int fd, const char *buf, size_t size)
{
	off_t offset = 0;
	size_t rem = size;
	ssize_t ret;

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

void CTFAPI::CTFStream::flushData()
{
	//TODO go async

	size_t size;

	size = head - tailCommited;

	doWrite(fdOutput, buffer + (tailCommited & mask), size);

	tail = head;
	tailCommited = head;
}
