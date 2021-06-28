/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <cinttypes>

#include "CTFMetadata.hpp"
#include "CTFTrace.hpp"
#include "CTFTypes.hpp"
#include "InstrumentExternalThreadLocalData.hpp"
#include "executors/threads/CPUManager.hpp"


std::string CTFAPI::CTFMetadata::_cpuList;
uint32_t CTFAPI::CTFMetadata::_externalThreadsCount = 0;
const char *CTFAPI::CTFMetadata::_meta_commonEnv =
	"	/* ctf2prv converter variables */\n"
	"	nanos6_trace_version = %d;\n"
	"	cpu_list = \"%s\";\n"
	"	external_thread_count = %" PRIu32 "; // LT + ETs\n"
	"	binary_name = \"%s\";\n"
	"	rank = \"%" PRIu32 "\";\n"
	"	nranks = \"%" PRIu32 "\";\n"
	"	pid = %" PRIu64 ";\n"
	"	start_ts = %" PRIu64 "; // (ns w/o correction)\n"
	"	end_ts = %" PRIu64 "; // (ns w/o correction)\n"
	"	time_correction = %" PRId64 "; // (ns)\n"
	"};\n\n";


void CTFAPI::CTFMetadata::collectCommonInformationAtInit()
{
	std::vector<CPU *> cpus = CPUManager::getCPUListReference();
	ctf_cpu_id_t totalCPUs = (ctf_cpu_id_t) cpus.size();

	ctf_cpu_id_t cpuId = cpus[0]->getSystemCPUId();
	_cpuList = std::to_string(cpuId);
	for (ctf_cpu_id_t i = 1; i < totalCPUs; i++) {
		cpuId = cpus[i]->getSystemCPUId();
		_cpuList += "," + std::to_string(cpuId);
	}
}

void CTFAPI::CTFMetadata::collectCommonInformationAtShutdown()
{
	_externalThreadsCount = Instrument::ExternalThreadLocalData::getExternalThreadCount();
}

void CTFAPI::CTFMetadata::printCommonMetaEnv(FILE *f)
{
	CTFTrace &trace = CTFTrace::getInstance();

	fprintf(f, _meta_commonEnv,
		CTFTrace::getTraceVersion(),
		_cpuList.c_str(),
		_externalThreadsCount,
		trace.getBinaryName(),
		trace.getRank(),
		trace.getNumberOfRanks(),
		trace.getPid(),
		trace.getAbsoluteStartTimestamp(),
		trace.getAbsoluteEndTimestamp(),
		trace.getTimeCorrection()
	);
}
