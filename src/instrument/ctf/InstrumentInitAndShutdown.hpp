/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_NULL_INIT_AND_SHUTDOWN_HPP


#include "CTFAPI.hpp"
#include <executors/threads/CPUManager.hpp>
#include "../api/InstrumentInitAndShutdown.hpp"


namespace Instrument {
	void initialize()
	{
		uint64_t i;
		uint64_t totalCPUs;
		size_t const defaultSize = 4096;

		CTFAPI::tracepoint();

		totalCPUs = (uint64_t) CPUManager::getTotalCPUs();

		for (i = 0; i < totalCPUs; i++) {
			CPU *CPU = CPUManager::getCPU(i);
			if (CPU) {
				CPULocalData &data = CPU->getInstrumentationData();
				data.initialize(defaultSize);
			}
		}
	}

	void shutdown()
	{
		uint64_t i;
		uint64_t totalCPUs;

		CTFAPI::tracepoint();

		totalCPUs = (uint64_t) CPUManager::getTotalCPUs();

		for (i = 0; i < totalCPUs; i++) {
			CPU *CPU = CPUManager::getCPU(i);
			if (CPU) {
				CPULocalData &data = CPU->getInstrumentationData();
				data.shutdown();
			}
		}
	}
}


#endif // INSTRUMENT_NULL_INIT_AND_SHUTDOWN_HPP
