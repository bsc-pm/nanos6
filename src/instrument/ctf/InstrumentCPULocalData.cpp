#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sys/mman.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <lowlevel/FatalErrorHandler.hpp>
#include <InstrumentCPULocalData.hpp>

#define PAGE_SHIFT (12)
#define PAGE_SIZE (1 << PAGE_SHIFT)



void Instrument::CTFStream::initialize(size_t size)
{
	int fd;
	size_t nPages;
	void *mrbPart;
	size_t sizeAligned;

	nPages = (size + (PAGE_SIZE - 1)) >> PAGE_SHIFT;
	sizeAligned = nPages * PAGE_SIZE;
	mrbSize = sizeAligned * 2;

	// allocate backing memory for the event buffer
	fd = open("/tmp", O_TMPFILE | O_RDWR | O_EXCL, 0600);
	if (fd == -1) {
		FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: open: ") + strerror(errno));
	}
	if (ftruncate(fd, sizeAligned)) {
		FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: ftruncate: ") + strerror(errno));
	}
	mrb = mmap(NULL, mrbSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (mrb == MAP_FAILED) {
		FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: mmap base: ") + strerror(errno));
	}

	// mmap magic ring buffer
	mrbPart = mmap(mrb, sizeAligned, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, fd, 0);
	if (mrbPart == MAP_FAILED) {
		FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: mmap part 1: ") + strerror(errno));
	}
	mrbPart = mmap((char *) mrb + sizeAligned, sizeAligned, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, fd, sizeAligned);
	if (mrbPart == MAP_FAILED) {
		FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: mmap part 2: ") + strerror(errno));
	}

	close(fd);

	// set initial values
	lost = 0;
	head = 0;
	tail = 0;
	threshold = 256*1024*1024; // TODO read this from env var
	tailCommited = 0;
	mask = sizeAligned - 1;
	buffer = (char *) mrb;
	bufferSize = sizeAligned;
	fileOffset = 0;
}

void Instrument::CTFStream::shutdown(void)
{
	if (!bufferSize)
		return;

	// TODO flush remainig buffer here ?
	munmap(mrb, mrbSize);
	head = 0;
	tail = 0;
	tailCommited = 0;
	bufferSize = 0;
}

bool Instrument::CTFStream::checkFreeSpace(size_t size)
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

void Instrument::CTFStream::doWrite(int fd, const char *buf, size_t size)
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
		fileOffset += ret;
		rem        -= ret;
	} while (rem > 0);
}

void Instrument::CTFStream::flushData()
{
	//TODO go async

	size_t size;

	size = head - tail;

	doWrite(fdOutput, buffer, size);

	tail = head;
	tailCommited = head;
}
