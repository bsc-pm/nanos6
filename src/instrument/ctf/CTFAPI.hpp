/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CTFAPI_HPP
#define CTFAPI_HPP

#include <stdint.h>
#include <inttypes.h>
#include <InstrumentCPULocalData.hpp>

namespace CTFAPI {
	void tracepoint(void);
	void tp_task_start(uint64_t addr, uint64_t taskId);
	void tp_task_stop(uint64_t taskId);

	void writeMetadata(void);
	void addStreamHeader(Instrument::CTFStream &stream);
	void writeUserMetadata(std::string directory);
	void writeKernelMetadata(std::string directory);
}

#endif // CTFAPI_HPP
