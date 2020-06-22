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
#include "ctfapi/context/CTFStreamContextUnbounded.hpp"
#include "CTFTracepoints.hpp"
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
	CTFAPI::CTFEventContext *ctfContextTaskHWC = nullptr;
	CTFAPI::CTFEventContext *ctfContextCPUHWC = nullptr;

	// Initialize Contexes
	if (HardwareCounters::hardwareCountersEnabled()) {
		ctfContextTaskHWC = userMetadata->addContext(
			new CTFAPI::CTFContextTaskHardwareCounters(CTFAPI::CTFStreamBoundedId)
		);
		ctfContextCPUHWC  = userMetadata->addContext(
			new CTFAPI::CTFContextCPUHardwareCounters(CTFAPI::CTFStreamBoundedId)
		);
	}

	// Add Contexes to evens that support them
	std::set<CTFAPI::CTFEvent *> &events = userMetadata->getEvents();
	for (auto it = events.begin(); it != events.end(); ++it) {
		CTFAPI::CTFEvent *event = (*it);
		uint8_t enabledContexes = event->getEnabledContexes();
		if (enabledContexes & CTFAPI::CTFContextTaskHWC) {
			event->addContext(ctfContextTaskHWC);
		} else if (enabledContexes & CTFAPI::CTFContextRuntimeHWC) {
			event->addContext(ctfContextCPUHWC);
		}
	}
}

static void initializeCTFBuffers(CTFAPI::CTFMetadata *userMetadata, std::string userPath)
{
	CPU *cpu;
	ctf_cpu_id_t cpuId;
	std::vector<CPU *> cpus = CPUManager::getCPUListReference();
	ctf_cpu_id_t totalCPUs = (ctf_cpu_id_t) cpus.size();

	const size_t defaultBufferSize = 2*1024*1024;
	//const size_t defaultBufferSize = 4096;
	//std::cout << "WARNING: buffer size set to " << defaultBufferSize << std::endl;

	//TODO init kernel stream

	// create and register contexes for streams
	CTFAPI::CTFStreamContextUnbounded *context = userMetadata->addContext(
		new CTFAPI::CTFStreamContextUnbounded()
	);

	// Initialize Worker thread streams
	for (ctf_cpu_id_t i = 0; i < totalCPUs; i++) {
		cpuId = i;
		cpu = cpus[i];
		Instrument::CPULocalData &cpuLocalData = cpu->getInstrumentationData();
		cpuLocalData.userStream = new CTFAPI::CTFStream(
			defaultBufferSize, cpuId, userPath.c_str()
		);
	}

	// Initialize Leader Thread Stream
	cpuId = totalCPUs;
	cpu = CPUManager::getLeaderThreadCPU();
	Instrument::CPULocalData &leaderThreadCPULocalData = cpu->getInstrumentationData();
	CTFAPI::CTFStreamUnboundedPrivate *unboundedPrivateStream = new CTFAPI::CTFStreamUnboundedPrivate(
		defaultBufferSize, cpuId, userPath.c_str()
	);
	unboundedPrivateStream->addContext(context);
	leaderThreadCPULocalData.userStream = unboundedPrivateStream;

	// Initialize External Threads Stream
	cpuId = totalCPUs + 1;
	Instrument::virtualCPULocalData = new Instrument::CPULocalData();
	CTFAPI::CTFStreamUnboundedShared *unboundedSharedStream = new CTFAPI::CTFStreamUnboundedShared(
		defaultBufferSize, cpuId, userPath.c_str()
	);
	unboundedSharedStream->addContext(context);
	Instrument::virtualCPULocalData->userStream = unboundedSharedStream;
}

void Instrument::initialize()
{
	std::string userPath, kernelPath;

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
	CPU *cpu;
	std::vector<CPU *> cpus = CPUManager::getCPUListReference();
	ctf_cpu_id_t totalCPUs = (ctf_cpu_id_t) cpus.size();
	CTFAPI::CTFTrace &trace = CTFAPI::CTFTrace::getInstance();

	// Shutdown Worker thread streams
	for (ctf_cpu_id_t i = 0; i < totalCPUs; i++) {
		cpu = cpus[i];
		CTFAPI::CTFStream *userStream = cpu->getInstrumentationData().userStream;
		userStream->shutdown();
		delete userStream;
	}

	// Shutdown Leader thread stream
	cpu = CPUManager::getLeaderThreadCPU();
	Instrument::CPULocalData &leaderThreadCPULocalData = cpu->getInstrumentationData();
	CTFAPI::CTFStream *leaderThreadStream = leaderThreadCPULocalData.userStream;
	leaderThreadStream->shutdown();
	delete leaderThreadStream;

	// Shutdown External thread stream
	CTFAPI::CTFStream *externalThreadStream = Instrument::virtualCPULocalData->userStream;
	externalThreadStream->shutdown();
	delete externalThreadStream;
	delete Instrument::virtualCPULocalData;

	// move tracing files to final directory
	trace.convertToParaver();
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
