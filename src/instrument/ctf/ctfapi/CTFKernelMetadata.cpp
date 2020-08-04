/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/


#include <errno.h>
#include <sys/utsname.h>

#include <cinttypes>
#include <cstdint>
#include <fstream>

#include "CTFTrace.hpp"
#include "CTFKernelMetadata.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "stream/CTFStream.hpp"
#include "support/JsonFile.hpp"


// TODO factor out common operations/data with CTFMetadata

const char *CTFAPI::CTFKernelMetadata::defaultKernelDefsFileName    = "./nanos6_kerneldefs.json";

const char *CTFAPI::CTFKernelMetadata::defaultEnabledEventsFileName = "./nanos6_kernel_events.txt";

const char *CTFAPI::CTFKernelMetadata::meta_header = "/* CTF 1.8 */\n";

const char *CTFAPI::CTFKernelMetadata::meta_typedefs =
	"typealias integer { size = 8; align = 8; signed = false; }  := uint8_t;\n"
	"typealias integer { size = 16; align = 8; signed = false; } := uint16_t;\n"
	"typealias integer { size = 32; align = 8; signed = false; } := uint32_t;\n"
	"typealias integer { size = 64; align = 8; signed = false; } := uint64_t;\n"
	"typealias floating_point { exp_dig =  8; mant_dig = 24; byte_order = native; align = 8; } := float;\n"
	"typealias floating_point { exp_dig = 11; mant_dig = 53; byte_order = native; align = 8; } := double;\n"
	"\n";

const char *CTFAPI::CTFKernelMetadata::meta_trace =
	"trace {\n"
	"	major = 1;\n"
	"	minor = 8;\n"
	"	byte_order = le;\n"
	"	packet.header := struct {\n"
	"		uint32_t magic;\n"
	"		uint32_t stream_id;\n"
	"	};\n"
	"};\n\n";

const char *CTFAPI::CTFKernelMetadata::meta_env =
	"env {\n"
	"	/* Trace Compass variables */\n"
	"	domain = \"kernel\";\n"
	"	sysname = \"Linux\";\n"
	"	kernel_release = \"%s\";\n"
	"	kernel_version = \"%s\";\n"
	"	tracer_name = \"lttng-modules\";\n"
	"	tracer_major = 2;\n"
	"	tracer_minor = 10;\n"
	"	tracer_patchlevel = 10;\n"
	"	/* ctf2prv converter variables */\n"
	"	ncpus = %" PRIu16 ";\n"
	"	binary_name = \"%s\";\n"
	"	pid = %" PRIu64 ";\n"
	"};\n\n";

const char *CTFAPI::CTFKernelMetadata::meta_clock =
	"clock {\n"
	"	name = \"monotonic\";\n"
	"	description = \"Monotonic Clock\";\n"
	"	freq = 1000000000; /* Frequency, in Hz */\n"
	"	/* clock value offset from Epoch is: offset * (1/freq) */\n"
	"	offset = %" PRIu64 ";\n"
	"};\n"
	"\n"
	"typealias integer {\n"
	"	size = 64;\n"
	"	align = 8;\n"
	"	signed = false;\n"
	"	map = clock.monotonic.value;\n"
	"} := uint64_clock_monotonic_t;\n\n";

const char *CTFAPI::CTFKernelMetadata::meta_stream =
	"stream {\n"
	"	id = %d;\n"
	"	packet.context := struct {\n"
	"		uint16_t cpu_id;\n"
	"	};\n"
	"	event.header := struct {\n"
	"		uint16_t id;\n"
	"		uint64_clock_monotonic_t timestamp;\n"
	"	};\n"
	"};\n\n";

const char *CTFAPI::CTFKernelMetadata::meta_event =
	"event {\n"
	"	name = \"%s\";\n"
	"	id = %d;\n"
	"	stream_id = %d;\n"
	"	fields := struct {\n"
	"%s"
	"	};\n"
	"};\n";

void CTFAPI::CTFKernelMetadata::getSystemInformation()
{
	int ret;
	struct utsname info;

	ret = uname(&info);
	if (ret) {
		FatalErrorHandler::fail(
			"CTF: Kernel: When calling uname: ",
			strerror(errno)
		);
	}

	_kernelVersion = std::string(info.version);
	_kernelRelease = std::string(info.release);
}

bool CTFAPI::CTFKernelMetadata::loadKernelDefsFile(const char *file)
{
	bool success = false;
	JsonFile configFile = JsonFile(file);

	if (configFile.fileExists()) {
		configFile.loadData();

		// Navigate through the file and extract tracepoint definitions
		configFile.getRootNode()->traverseChildrenNodes(
			[&](const std::string &name, const JsonNode<> &node) {
				if (name == "meta") {
					bool converted = false;
					// get number of events
					_numberOfEvents = node.getData<ctf_kernel_event_id_t>("numberOfEvents", converted);
					assert(converted);

					// get max event id
					_maxEventId = node.getData<ctf_kernel_event_id_t>("maxEventId", converted);
					assert(converted);

					_eventSizes.reserve(_maxEventId + 1);
				} else {
					// get the event ID
					bool converted = false;
					ctf_kernel_event_id_t id = node.getData<ctf_kernel_event_id_t>("id", converted);
					assert(converted);

					// get the event Size
					ctf_kernel_event_size_t size = node.getData<ctf_kernel_event_size_t>("size", converted);
					assert(converted);

					// get the event Format
					std::string format = node.getData<std::string>("format", converted);
					assert(converted);

					_idMap.emplace(name, std::make_pair(id, format));
					_eventSizes[id] = size;
				}
			}
		);

		success = (_maxEventId != (ctf_kernel_event_id_t) -1) && (_idMap.size() > 0);

		FatalErrorHandler::failIf(!success, "CTF: kernel: Kernel events definitions file present, but corrupted");
	}

	return success;
}

std::string trim(
	const std::string& str,
	const std::string& whitespace = " \t"
) {
	const auto strBegin = str.find_first_not_of(whitespace);
	if (strBegin == std::string::npos)
		return ""; // no content

	const auto strEnd = str.find_last_not_of(whitespace);
	const auto strRange = strEnd - strBegin + 1;

	return str.substr(strBegin, strRange);
}

bool CTFAPI::CTFKernelMetadata::loadEnabledEvents(const char *file)
{
	std::string line;
	std::ifstream streamFile;

	streamFile.open (file);

	if (!streamFile.is_open()) {
		return false;
	}

	while (std::getline(streamFile, line)) {
		std::string trimmedLine = trim(line);

		if (trimmedLine[0] == '#' || trimmedLine == "")
			continue;

		try {
			auto const& entry = _idMap.at(trimmedLine);
			ctf_kernel_event_id_t id = entry.first;
			_enabledEventIds.push_back(id);
			_enabledEventNames.push_back(trimmedLine);
		} catch(std::out_of_range &e) {
			FatalErrorHandler::fail("CTF: Kernel: The event \"", line, "\" is not found into the kernel tracepoint definition file");
		}
	}

	streamFile.close();

	FatalErrorHandler::warnIf(
		_enabledEventIds.size() == 0,
		"CTF: Kernel: Kernel tracepoints file found, but no event enabled."
	);

	return _enabledEventIds.size() > 0;
}

CTFAPI::CTFKernelMetadata::CTFKernelMetadata()
	: _enabled(false), _maxEventId(-1)
{
	bool loadDefs, loadEvents;

	getSystemInformation();
	loadDefs = loadKernelDefsFile(defaultKernelDefsFileName);
	loadEvents = loadEnabledEvents(defaultEnabledEventsFileName);

	if (loadDefs && loadEvents) {
		_enabled = true;
	}
}

void CTFAPI::CTFKernelMetadata::writeMetadataFile(std::string kernelPath)
{
	int ret;
	FILE *f;
	std::string path;

	if (!_enabled)
		return;

	CTFTrace &trace = CTFTrace::getInstance();

	path = kernelPath + "/metadata";

	f = fopen(path.c_str(), "w");
	if (f == NULL) {
		FatalErrorHandler::fail(
			"Instrumentation: ctf: writting kernel metadata file: ",
			strerror(errno)
		);
	}

	fputs(meta_header, f);
	fputs(meta_typedefs, f);
	fputs(meta_trace, f);
	fprintf(f, meta_env,
		_kernelRelease.c_str(),
		_kernelVersion.c_str(),
		trace.getTotalCPUs(),
		trace.getBinaryName(),
		trace.getPid());
	fprintf(f, meta_clock, trace.getAbsoluteStartTimestamp());
	fprintf(f, meta_stream, CTFStreamKernelId);

	for (std::string event : _enabledEventNames) {
		auto const& entry = _idMap.at(event);
		fprintf(f, meta_event,
			event.c_str(),       // event name
			entry.first,         // event id
			CTFStreamKernelId,   // stream id
			entry.second.c_str() // event format
		);
	}

	ret = fclose(f);
	if (ret) {
		FatalErrorHandler::warn("CTF: Kernel: closing metadata file: ", strerror(errno));
	}
}


void CTFAPI::CTFKernelMetadata::copyKernelDefinitionsFile(std::string basePath)
{
	if (!_enabled)
		return;

	std::ifstream  src(defaultKernelDefsFileName, std::ios::binary);
	std::ofstream  dst(basePath + "/" + defaultKernelDefsFileName, std::ios::binary);

	dst << src.rdbuf();
}
