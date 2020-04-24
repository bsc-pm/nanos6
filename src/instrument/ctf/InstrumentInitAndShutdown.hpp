/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_INIT_AND_SHUTDOWN_HPP
#define INSTRUMENT_CTF_INIT_AND_SHUTDOWN_HPP

#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "executors/threads/CPUManager.hpp"
#include "../api/InstrumentInitAndShutdown.hpp"

#include "ctfapi/CTFTypes.hpp"
#include "ctfapi/CTFAPI.hpp"
#include "ctfapi/CTFTrace.hpp"
#include "ctfapi/CTFMetadata.hpp"
#include "ctfapi/CTFStream.hpp"
#include "Nanos6CTFEvents.hpp"


namespace Instrument {

	static void refineCTFEvents(__attribute__((unused)) CTFAPI::CTFMetadata *metadata)
	{
		// TODO perform refinement based on the upcoming Nanos6 JSON
		// TODO add custom user-defined events based JSON
	}

	static void initializeCTFEvents(CTFAPI::CTFMetadata *userMetadata)
	{
		// create event Contexes
		CTFAPI::CTFContext *ctfContextHWC = userMetadata->addContext(new CTFAPI::CTFContextHardwareCounters());

		std::set<CTFAPI::CTFEvent *> &events = userMetadata->getEvents();
		for (auto it = events.begin(); it != events.end(); ++it) {
			CTFAPI::CTFEvent *event = (*it);
			uint8_t enabledContexes = event->getEnabledContexes();
			if (enabledContexes & CTFAPI::CTFContextHWC)
				event->addContext(ctfContextHWC);
		}
	}

	static void initializeCTFBuffers(std::string userPath)
	{
		ctf_cpu_id_t i;
		ctf_cpu_id_t cpuId;
		std::string streamPath;
		const size_t defaultSize = 4096;
		ctf_cpu_id_t totalCPUs = (ctf_cpu_id_t) CPUManager::getTotalCPUs();

		// TODO allocate memory on each CPU (madvise or specific
		// instrument function?)

		for (i = 0; i < totalCPUs; i++) {
			CPU *CPU = CPUManager::getCPU(i);
			assert(CPU != nullptr);
			cpuId = CPU->getSystemCPUId();
			CPULocalData &cpuLocalData = CPU->getInstrumentationData();

			//TODO init kernel stream

			CTFAPI::CTFStream *userStream = new CTFAPI::CTFStream;
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
		CTFAPI::ExclusiveCTFStream *exclusiveUserStream = new CTFAPI::ExclusiveCTFStream;
		exclusiveUserStream->initialize(defaultSize, cpuId);
		CTFAPI::addStreamHeader(exclusiveUserStream);
		streamPath = userPath + "/channel_" + std::to_string(cpuId);
		exclusiveUserStream->fdOutput = open(streamPath.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0666);
		if (exclusiveUserStream->fdOutput == -1)
			FatalErrorHandler::failIf(true, std::string("Instrument: ctf: failed to open stream file: ") + strerror(errno));
		virtualCPULocalData->userStream = exclusiveUserStream;
	}

	void initialize()
	{
		std::string userPath, kernelPath;

		CTFAPI::greetings();
		CTFAPI::CTFTrace &trace = CTFAPI::CTFTrace::getInstance();
		CTFAPI::CTFMetadata *userMetadata = new CTFAPI::CTFMetadata();

		trace.setMetadata(userMetadata);
		trace.setTotalCPUs(CPUManager::getTotalCPUs());
		trace.setTracePath("./trace-ctf-nanos6");
		trace.initializeTraceTimer();
		trace.createTraceDirectories(userPath, kernelPath);
		initializeCTFBuffers(userPath);

		preinitializeCTFEvents(userMetadata);
		refineCTFEvents(userMetadata);
		initializeCTFEvents(userMetadata);
		userMetadata->writeMetadataFile(userPath);
	}

	void shutdown()
	{
		ctf_cpu_id_t i;
		ctf_cpu_id_t totalCPUs;
		CTFAPI::CTFTrace &trace = CTFAPI::CTFTrace::getInstance();

		CTFAPI::greetings();

		totalCPUs = (ctf_cpu_id_t) CPUManager::getTotalCPUs();

		for (i = 0; i < totalCPUs; i++) {
			CPU *CPU = CPUManager::getCPU(i);
			assert(CPU != nullptr);

			CTFAPI::CTFStream *userStream = CPU->getInstrumentationData().userStream;
			userStream->flushData();

			if (userStream->lost)
				std::cerr << "WARNING: CTF Instrument: " << userStream->lost << " events lost in core " << i << std::endl;

			userStream->shutdown();
			close(userStream->fdOutput);
			delete userStream;
		}

		// TODO use true virtual cpu mechanism here
		CTFAPI::CTFStream *userStream = virtualCPULocalData->userStream;
		userStream->flushData();
		if (userStream->lost)
			std::cerr << "WARNING: CTF Instrument: " << userStream->lost << " events lost in core " << i << std::endl;
		userStream->shutdown();
		close(userStream->fdOutput);
		delete userStream;
		delete virtualCPULocalData;

		trace.clean();
	}
}


#endif // INSTRUMENT_CTF_INIT_AND_SHUTDOWN_HPP
