/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <cstdint>

class CircularBuffer {

private:
	char     *buffer;
	uint64_t bufferSize;
	uint64_t subBufferSize;
	uint64_t head;
	uint64_t tail;
	uint64_t wall;
	uint64_t hole;
	uint64_t mask;
	uint64_t subBufferMask;

	int fd;
	uint64_t fileOffset;

	void initializeFile(const char *path);
	void initializeBuffer(uint64_t size);
	void flushToFile(char *buf, size_t size);
	void flushUpToTheWrap();
	void resetPointers();

	bool wraps()
	{
		return ((head & ~mask) != (tail & ~mask));
	}

public:
	CircularBuffer() {};

	void initialize(uint64_t size, const char *path);
	void flushAll();
	void flushFilledSubBuffers();
	void shutdown();
	bool checkIfNeedsFlush();
	bool alloc(uint64_t size);

	inline void *getBuffer()
	{
		return (void *) (buffer + (head & mask));
	}

	inline void submit(uint64_t size)
	{
		head += size;
		assert(head - tail <= bufferSize);
	}

};
