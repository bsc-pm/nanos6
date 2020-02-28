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



bool Instrument::CPULocalData::initialize(size_t size)
{
	size_t nPages;
	void *mrbPart;
	int fd;
	size_t sizeAligned;
	bool ret = false;

	nPages = (size + (PAGE_SIZE - 1)) >> PAGE_SHIFT;
	sizeAligned = nPages * PAGE_SIZE;
	mrbSize = sizeAligned * 2;

	// allocate backing memory for the event buffer
	fd = open("/tmp", O_TMPFILE | O_RDWR | O_EXCL, S_IRUSR | S_IWUSR);
	if (fd == -1) {
		FatalErrorHandler::warnIf(true, std::string("Instrumentation: ctf: open: ") + strerror(errno));
		goto out;
	}
	if (ftruncate(fd, sizeAligned)) {
		FatalErrorHandler::warnIf(true, std::string("Instrumentation: ctf: ftruncate: ") + strerror(errno));
		goto close_fd;
	}
	mrb = mmap(NULL, mrbSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (mrb == MAP_FAILED) {
		FatalErrorHandler::warnIf(true, std::string("Instrumentation: ctf: mmap base: ") + strerror(errno));
		goto close_fd;
	}

	// mmap magic ring buffer
	mrbPart = mmap(mrb, sizeAligned, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, fd, 0);
	if (mrbPart == MAP_FAILED) {
		FatalErrorHandler::warnIf(true, std::string("Instrumentation: ctf: mmap part 1: ") + strerror(errno));
		munmap(mrb, mrbSize);
		goto close_fd;
	}
	mrbPart = mmap((char *) mrb + sizeAligned, sizeAligned, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED, fd, sizeAligned);
	if (mrbPart == MAP_FAILED) {
		FatalErrorHandler::warnIf(true, std::string("Instrumentation: ctf: mmap part 2: ") + strerror(errno));
		munmap(mrb, mrbSize);
		goto close_fd;
	}

	// set initial values
	head = 0;
	tail = 0;
	userEventBuffer = (char *) mrb;
	userEventBufferSize = sizeAligned;
	ret = true;

close_fd:
	close(fd);

out:
	return ret;
}

void Instrument::CPULocalData::shutdown(void)
{
	if (!userEventBufferSize)
		return;

	// TODO flush remainig buffer here ?
	munmap(mrb, mrbSize);
	head = 0;
	tail = 0;
}
