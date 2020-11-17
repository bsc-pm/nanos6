/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>
#include <cstdint>

class CircularBuffer {

private:
	char *_buffer;
	uint64_t _bufferSize;
	uint64_t _subBufferSize;
	uint64_t _head;
	uint64_t _tail;
	uint64_t _wall;
	uint64_t _hole;
	uint64_t _mask;
	uint64_t _subBufferMask;

	int _fd;
	uint64_t _fileOffset;
	int _node;

	void initializeFile(const char *path);
	void initializeBuffer(uint64_t size, int node);
	void flushToFile(char *buf, size_t size);
	void flushUpToTheWrap();
	void resetPointers();

	inline bool wraps()
	{
		return ((_head & ~_mask) != (_tail & ~_mask));
	}

public:
	CircularBuffer() {};

	void initialize(uint64_t size, int node, const char *path);
	void flushAll();
	void flushFilledSubBuffers();
	void shutdown();
	bool checkIfNeedsFlush();
	bool alloc(uint64_t size);
	uint64_t allocAtLeast(uint64_t minSize);

	inline void *getBuffer()
	{
		return (void *) (_buffer + (_head & _mask));
	}

	inline void submit(uint64_t size)
	{
		_head += size;
		assert(_head - _tail <= _bufferSize);
	}

};
