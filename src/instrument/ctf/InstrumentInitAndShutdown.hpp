/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_CTF_INIT_AND_SHUTDOWN_HPP

#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <InstrumentCPULocalData.hpp>
#include <executors/threads/CPUManager.hpp>
#include "../api/InstrumentInitAndShutdown.hpp"
#include "CTFAPI.hpp"


namespace Instrument {

	static void createTraceDirectories(std::string root, std::string &userPath, std::string &kernelPath)
	{
		int ret;

		userPath   = root;
		kernelPath = root;

		ret = mkdir(root.c_str(), 0766);
		FatalErrorHandler::failIf(ret, "Instrument: ctf: failed to create trace directories");

		kernelPath += "/kernel";
		ret = mkdir(kernelPath.c_str(), 0766);
		FatalErrorHandler::failIf(ret, "Instrument: ctf: failed to create trace directories");

		userPath += "/ust";
		ret = mkdir(userPath.c_str(), 0766);
		FatalErrorHandler::failIf(ret, "Instrument: ctf: failed to create trace directories");
		userPath += "/uid";
		ret = mkdir(userPath.c_str(), 0766);
		FatalErrorHandler::failIf(ret, "Instrument: ctf: failed to create trace directories");
		userPath += "/1042";
		ret = mkdir(userPath.c_str(), 0766);
		FatalErrorHandler::failIf(ret, "Instrument: ctf: failed to create trace directories");
		userPath += "/64-bit";
		ret = mkdir(userPath.c_str(), 0766);
		FatalErrorHandler::failIf(ret, "Instrument: ctf: failed to create trace directories");
	}

	void initialize()
	{
		bool ret;
		uint64_t i;
		uint64_t totalCPUs;
		uint32_t cpuId;
		struct timespec tp;
		const size_t defaultSize = 4096;
		const uint64_t ns = 1000000000ULL;
		std::string tracePath, userPath, kernelPath, streamPath;

		CTFAPI::greetings();

		if (clock_gettime(CLOCK_MONOTONIC, &tp)) {
			FatalErrorHandler::failIf(true, std::string("Instrumentation: ctf: initialize: clock_gettime syscall: ") + strerror(errno));
		}
		CTFAPI::core::absoluteStartTime = tp.tv_sec * ns + tp.tv_nsec;

		// TODO add timestamp?
		// TODO get folder name & path form env var?
		// TODO 1042 is the user id, get the real one
		// TODO allocate memory on each CPU (madvise or specific
		// instrument function?)
		tracePath = "./trace-ctf-nanos6";
		createTraceDirectories(tracePath, userPath, kernelPath);

		totalCPUs = (uint64_t) CPUManager::getTotalCPUs();
		CTFAPI::core::totalCPUs = totalCPUs;

		CTFAPI::writeUserMetadata(userPath);
		//CTFAPI::writeKernelMetadata(kernelPath);

		for (i = 0; i < totalCPUs; i++) {
			CPU *CPU = CPUManager::getCPU(i);
			assert(CPU != nullptr);
			cpuId = CPU->getSystemCPUId();
			CPULocalData &cpuLocalData = CPU->getInstrumentationData();

			//TODO init kernel stream

			CTFStream *userStream = new CTFStream;
			userStream->initialize(defaultSize, cpuId);
			CTFAPI::addStreamHeader(userStream);
			streamPath = userPath + "/channel_" + std::to_string(cpuId);
			userStream->fdOutput = open(streamPath.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0666);
			if (userStream->fdOutput == -1)
				FatalErrorHandler::failIf(true, std::string("Instrument: ctf: failed to open stream file: ") + strerror(errno));

			cpuLocalData.userStream = userStream;
		}

		// TODO use true virtual cpu mechanism here
		cpuId = totalCPUs;
		virtualCPULocalData = new CPULocalData();
		ExclusiveCTFStream *exclusiveUserStream = new ExclusiveCTFStream;
		exclusiveUserStream->initialize(defaultSize, cpuId);
		CTFAPI::addStreamHeader(exclusiveUserStream);
		streamPath = userPath + "/channel_" + std::to_string(cpuId);
		exclusiveUserStream->fdOutput = open(streamPath.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0666);
		if (exclusiveUserStream->fdOutput == -1)
			FatalErrorHandler::failIf(true, std::string("Instrument: ctf: failed to open stream file: ") + strerror(errno));
		virtualCPULocalData->userStream = exclusiveUserStream;
	}

	void shutdown()
	{
		uint64_t i;
		uint64_t totalCPUs;

		CTFAPI::greetings();

		totalCPUs = (uint64_t) CPUManager::getTotalCPUs();

		for (i = 0; i < totalCPUs; i++) {
			CPU *CPU = CPUManager::getCPU(i);
			assert(CPU != nullptr);

			CTFStream *userStream = CPU->getInstrumentationData().userStream;
			userStream->flushData();

			if (userStream->lost)
				std::cerr << "WARNING: CTF Instrument: " << userStream->lost << " events on core " << i << std::endl;

			userStream->shutdown();
			close(userStream->fdOutput);
			delete userStream;
		}

		// TODO use true virtual cpu mechanism here
		CTFStream *userStream = virtualCPULocalData->userStream;
		userStream->flushData();
		if (userStream->lost)
			std::cerr << "WARNING: CTF Instrument: " << userStream->lost << " events on core " << i << std::endl;
		userStream->shutdown();
		close(userStream->fdOutput);
		delete userStream;
		delete virtualCPULocalData;
	}
}


#endif // INSTRUMENT_CTF_INIT_AND_SHUTDOWN_HPP
