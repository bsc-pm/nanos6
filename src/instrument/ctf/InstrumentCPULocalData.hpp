/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CPU_LOCAL_DATA_HPP
#define INSTRUMENT_CPU_LOCAL_DATA_HPP

namespace Instrument {
	// TODO CPUlocalData or CPULocalData_t ?
	struct CPULocalData {
		char *userEventBuffer;
		size_t userEventBufferSize;
		unsigned long long int head;
		unsigned long long int tail;

		CPULocalData() : userEventBufferSize(0) {}
		bool initialize(size_t size);
		void shutdown(void);

	private:
		void *mrb;
		size_t mrbSize;
	};
}

#endif //INSTRUMENT_CPU_LOCAL_DATA_HPP
