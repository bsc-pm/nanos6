/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "hardware-counters/HardwareCounters.hpp"

#include "ctfapi/CTFTypes.hpp"
#include "ctfapi/CTFAPI.hpp"
#include "ctfapi/CTFTrace.hpp"
#include "ctfapi/CTFMetadata.hpp"
#include "ctfapi/stream/CTFStream.hpp"
#include "ctfapi/stream/CTFStreamUnboundedPrivate.hpp"
#include "ctfapi/stream/CTFStreamUnboundedShared.hpp"
#include "ctfapi/context/CTFContextTaskHardwareCounters.hpp"
#include "ctfapi/context/CTFContextCPUHardwareCounters.hpp"
#include "ctfapi/context/CTFContextUnbounded.hpp"
#include "tracepoints.hpp"
#include "tasks/TaskInfo.hpp"
#include "tasks/TasktypeData.hpp"
#include "executors/threads/CPUManager.hpp"

#include "InstrumentInitAndShutdown.hpp"


static void refineCTFEvents(__attribute__((unused)) CTFAPI::CTFMetadata *metadata)
{
	// TODO perform refinement based on the upcoming Nanos6 JSON
	// TODO add custom user-defined events based JSON
}

static void initializeCTFEvents(CTFAPI::CTFMetadata *userMetadata)
{
	CTFAPI::CTFContext *ctfContextTaskHWC = nullptr;
	CTFAPI::CTFContext *ctfContextCPUHWC = nullptr;

	// Initialize Contexes
	if (HardwareCounters::hardwareCountersEnabled()) {
		ctfContextTaskHWC = userMetadata->addContext(new CTFAPI::CTFContextTaskHardwareCounters());
		ctfContextCPUHWC  = userMetadata->addContext(new CTFAPI::CTFContextCPUHardwareCounters());
	}

	// Add Contexes to evens that support them
	std::set<CTFAPI::CTFEvent *> &events = userMetadata->getEvents();
	for (auto it = events.begin(); it != events.end(); ++it) {
		CTFAPI::CTFEvent *event = (*it);
		uint8_t enabledContexes = event->getEnabledContexes();
		if (enabledContexes & CTFAPI::CTFContextTaskHWC) {
			event->addContext(ctfContextTaskHWC);
		} else if (enabledContexes & CTFAPI::CTFContextCPUHWC) {
			event->addContext(ctfContextCPUHWC);
		}
	}
}

static void initializeCTFBuffers(CTFAPI::CTFMetadata *userMetadata, std::string userPath)
{
	CPU *CPU;
	ctf_cpu_id_t i;
	ctf_cpu_id_t cpuId;
	std::string streamPath;
	const size_t defaultBufferSize = 2*1024*1024;
	//const size_t defaultBufferSize = 4096;
	//std::cout << "WARNING: buffer size set to " << defaultBufferSize << std::endl;
	ctf_cpu_id_t totalCPUs = (ctf_cpu_id_t) CPUManager::getTotalCPUs();

	assert(defaultBufferSize % 2 == 0);

	// TODO can we place this initialization code somewhere else?
	// maybe under CTFAPI or CTFTrace?
	//
	// TODO allocate memory on each CPU (madvise or specific
	// instrument function?)

	for (i = 0; i < totalCPUs; i++) {
		CPU = CPUManager::getCPU(i);
		cpuId = CPU->getSystemCPUId();
		Instrument::CPULocalData &cpuLocalData = CPU->getInstrumentationData();

		//TODO init kernel stream

		CTFAPI::CTFStream *userStream = new CTFAPI::CTFStream;
		userStream->initialize(defaultBufferSize, cpuId);
		CTFAPI::addStreamHeader(userStream);
		streamPath = userPath + "/channel_" + std::to_string(cpuId);
		userStream->fdOutput = open(streamPath.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0666);
		if (userStream->fdOutput == -1)
			FatalErrorHandler::failIf(true, std::string("Instrument: ctf: failed to open stream file: ") + strerror(errno));

		cpuLocalData.userStream = userStream;
	}

	CTFAPI::CTFContextUnbounded *context = new CTFAPI::CTFContextUnbounded();
	userMetadata->addContext(context);

	cpuId = totalCPUs;
	CPU = CPUManager::getLeaderThreadCPU();
	Instrument::CPULocalData &leaderThreadCPULocalData = CPU->getInstrumentationData();
	CTFAPI::CTFStreamUnboundedPrivate *unboundedPrivateStream = new CTFAPI::CTFStreamUnboundedPrivate();
	unboundedPrivateStream->setContext(context);
	unboundedPrivateStream->initialize(defaultBufferSize, cpuId);
	CTFAPI::addStreamHeader(unboundedPrivateStream);
	streamPath = userPath + "/channel_" + std::to_string(cpuId);
	unboundedPrivateStream->fdOutput = open(streamPath.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0666);
	if (unboundedPrivateStream->fdOutput == -1)
		FatalErrorHandler::failIf(true, std::string("Instrument: ctf: failed to open stream file: ") + strerror(errno));
	leaderThreadCPULocalData.userStream = unboundedPrivateStream;

	cpuId = totalCPUs + 1;
	Instrument::virtualCPULocalData = new Instrument::CPULocalData();
	CTFAPI::CTFStreamUnboundedShared *unboundedSharedStream = new CTFAPI::CTFStreamUnboundedShared();
	unboundedSharedStream->setContext(context);
	unboundedSharedStream->initialize(defaultBufferSize, cpuId);
	CTFAPI::addStreamHeader(unboundedSharedStream);
	streamPath = userPath + "/channel_" + std::to_string(cpuId);
	unboundedSharedStream->fdOutput = open(streamPath.c_str(), O_WRONLY | O_TRUNC | O_CREAT, 0666);
	if (unboundedSharedStream->fdOutput == -1)
		FatalErrorHandler::failIf(true, std::string("Instrument: ctf: failed to open stream file: ") + strerror(errno));
	Instrument::virtualCPULocalData->userStream = unboundedSharedStream;
}

void Instrument::initialize()
{
	std::string userPath, kernelPath;

	CTFAPI::greetings();
	CTFAPI::CTFTrace &trace = CTFAPI::CTFTrace::getInstance();
	CTFAPI::CTFMetadata *userMetadata = new CTFAPI::CTFMetadata();

	trace.setMetadata(userMetadata);
	trace.setTracePath(".");
	trace.initializeTraceTimer();
	trace.setTotalCPUs(CPUManager::getTotalCPUs());
	trace.createTraceDirectories(userPath, kernelPath);
	initializeCTFBuffers(userMetadata, userPath);

	preinitializeCTFEvents(userMetadata);
	refineCTFEvents(userMetadata);
	initializeCTFEvents(userMetadata);
	userMetadata->writeMetadataFile(userPath);
}

void Instrument::shutdown()
{
	CPU *CPU;
	ctf_cpu_id_t i;
	ctf_cpu_id_t totalCPUs;
	CTFAPI::CTFTrace &trace = CTFAPI::CTFTrace::getInstance();

	CTFAPI::greetings();

	totalCPUs = (ctf_cpu_id_t) CPUManager::getTotalCPUs();

	for (i = 0; i < totalCPUs; i++) {
		CPU = CPUManager::getCPU(i);

		CTFAPI::CTFStream *userStream = CPU->getInstrumentationData().userStream;
		userStream->flushAll();

		userStream->shutdown();
		close(userStream->fdOutput);
		delete userStream;
	}

	CPU = CPUManager::getLeaderThreadCPU();
	Instrument::CPULocalData &leaderThreadCPULocalData = CPU->getInstrumentationData();
	CTFAPI::CTFStream *leaderThreadStream = leaderThreadCPULocalData.userStream;
	leaderThreadStream->flushAll();
	leaderThreadStream->shutdown();
	close(leaderThreadStream->fdOutput);
	delete leaderThreadStream;

	CTFAPI::CTFStream *externalThreadStream = Instrument::virtualCPULocalData->userStream;
	externalThreadStream->flushAll();
	externalThreadStream->shutdown();
	close(externalThreadStream->fdOutput);
	delete externalThreadStream;
	delete Instrument::virtualCPULocalData;

	trace.moveTemporalTraceToFinalDirectory();
	trace.clean();
}

void Instrument::nanos6_preinit_finished()
{
	// emit an event per each registered task type with its label and source
	TaskInfo::processAllTasktypes(
		[&](const std::string &tasktypeLabel, const std::string &tasktypeSource, TasktypeData &tasktypeData) {
			TasktypeInstrument &instrumentId = tasktypeData.getInstrumentationId();
			ctf_tasktype_id_t tasktypeId = instrumentId.autoAssingId();
			tp_task_label(tasktypeLabel.c_str(), tasktypeSource.c_str(), tasktypeId);
		}
	);
}
