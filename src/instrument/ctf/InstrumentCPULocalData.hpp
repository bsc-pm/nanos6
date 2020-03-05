/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CPU_LOCAL_DATA_HPP
#define INSTRUMENT_CPU_LOCAL_DATA_HPP

#include <string>
#include <stdint.h>

namespace Instrument {
	// TODO CPUlocalData or CPULocalData_t ?
	struct CTFStream {
		char *buffer;
		size_t bufferSize;
		uint64_t head;
		uint64_t tail;
		uint64_t tailCommited;
		uint64_t mask;
		uint64_t lost;
		uint64_t threshold;
		uint32_t cpuId;

		int fdOutput;
		off_t fileOffset;

		void initialize(size_t size, uint32_t cpuId);
		void shutdown(void);
		bool checkFreeSpace(size_t size);
		void flushData();

		CTFStream() : bufferSize(0) {}

	private:
		void *mrb;
		size_t mrbSize;

		void doWrite(int fd, const char *buf, size_t size);
	};

	struct CPULocalData {
		CPULocalData() {}
		CTFStream userStream;
		CTFStream kernelStream;
	};

}

#endif //INSTRUMENT_CPU_LOCAL_DATA_HPP
