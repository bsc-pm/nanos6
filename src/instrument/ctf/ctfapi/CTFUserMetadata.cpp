/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2021 Barcelona Supercomputing Center (BSC)
*/

#include <fstream>
#include <cstdint>
#include <cstdio>
#include <cinttypes>
#include <vector>

#include "stream/CTFStream.hpp"
#include "CTFUserMetadata.hpp"
#include "context/CTFEventContext.hpp"
#include "CTFTrace.hpp"


const char *CTFAPI::CTFUserMetadata::meta_header = "/* CTF 1.8 */\n";

const char *CTFAPI::CTFUserMetadata::meta_typedefs =
	"typealias integer { size = 8; align = 8; signed = false; }  := uint8_t;\n"
	"typealias integer { size = 16; align = 8; signed = false; } := uint16_t;\n"
	"typealias integer { size = 32; align = 8; signed = false; } := uint32_t;\n"
	"typealias integer { size = 64; align = 8; signed = false; } := uint64_t;\n"
	"typealias floating_point { exp_dig =  8; mant_dig = 24; byte_order = native; align = 8; } := float;\n"
	"typealias floating_point { exp_dig = 11; mant_dig = 53; byte_order = native; align = 8; } := double;\n"
	"\n";

const char *CTFAPI::CTFUserMetadata::meta_trace =
	"trace {\n"
	"	major = 1;\n"
	"	minor = 8;\n"
	"	byte_order = le;\n"
	"	packet.header := struct {\n"
	"		uint32_t magic;\n"
	"		uint32_t stream_id;\n"
	"	};\n"
	"};\n\n";

const char *CTFAPI::CTFUserMetadata::meta_env =
	"env {\n"
	"	/* Trace Compass variables */\n"
	"	domain = \"ust\";\n"
	"	tracer_name = \"lttng-ust\";\n"
	"	tracer_major = 2;\n"
	"	tracer_minor = 11;\n"
	"	tracer_patchlevel = 0;\n";

const char *CTFAPI::CTFUserMetadata::meta_clock =
	"clock {\n"
	"	name = \"monotonic\";\n"
	"	description = \"Monotonic Clock\";\n"
	"	freq = 1000000000; /* Frequency, in Hz */\n"
	"	/* clock value offset from Epoch is: offset * (1/freq) */\n"
	"	offset = %" PRIi64 ";\n"
	"};\n"
	"\n"
	"typealias integer {\n"
	"	size = 64;\n"
	"	align = 8;\n"
	"	signed = false;\n"
	"	map = clock.monotonic.value;\n"
	"} := uint64_clock_monotonic_t;\n\n";

const char *CTFAPI::CTFUserMetadata::meta_streamBounded =
	"stream {\n"
	"	id = %d;\n"
	"	packet.context := struct {\n"
	"		uint16_t cpu_id;\n"
	"	};\n"
	"	event.header := struct {\n"
	"		uint8_t id;\n"
	"		uint64_clock_monotonic_t timestamp;\n"
	"	};\n"
	"};\n\n";

const char *CTFAPI::CTFUserMetadata::meta_streamUnbounded =
	"stream {\n"
	"	id = %d;\n"
	"	packet.context := struct {\n"
	"		uint16_t cpu_id;\n"
	"	};\n"
	"	event.header := struct {\n"
	"		uint8_t id;\n"
	"		uint64_clock_monotonic_t timestamp;\n"
	"	};\n"
	"	event.context := struct {\n"
	"		struct unbounded unbounded;\n"
	"	};\n"
	"};\n\n";

const char *CTFAPI::CTFUserMetadata::meta_eventMetadataId =
	"event {\n"
	"	name = \"%s\";\n"
	"	id = %d;\n";

const char *CTFAPI::CTFUserMetadata::meta_eventMetadataStreamId =
	"	stream_id = %d;\n";

const char *CTFAPI::CTFUserMetadata::meta_eventMetadataFields =
	"	fields := struct {\n"
	"%s"
	"	};\n"
	"};\n\n";


CTFAPI::CTFUserMetadata::~CTFUserMetadata()
{
	for (auto it = events.begin(); it != events.end(); ++it) {
		CTFAPI::CTFEvent *event = it->second;
		delete event;
	}
	for (auto p : contexes)
		delete p;
	events.clear();
	contexes.clear();
}

void CTFAPI::CTFUserMetadata::writeEventContextMetadata(FILE *f, CTFAPI::CTFEvent *event, ctf_stream_id_t streamId)
{
	std::vector<CTFAPI::CTFEventContext *> &eventContexes = event->getContexes();
	if (eventContexes.empty())
		return;

	fprintf(f, "\tcontext := struct {\n");
	for (auto it = eventContexes.begin(); it != eventContexes.end(); ++it) {
		CTFAPI::CTFEventContext *context = (*it);
		if (context->getStreamMask() & streamId)
			fputs(context->getEventMetadata(), f);
	}
	fprintf(f, "\t};\n");
}

void CTFAPI::CTFUserMetadata::writeEventMetadata(FILE *f, CTFAPI::CTFEvent *event, ctf_stream_id_t streamId)
{
	fprintf(f, meta_eventMetadataId, event->getName(), event->getEventId());
	fprintf(f, meta_eventMetadataStreamId, streamId);
	writeEventContextMetadata(f, event, streamId);
	fprintf(f, meta_eventMetadataFields, event->getMetadataFields());
}

void CTFAPI::CTFUserMetadata::writeMetadataFile()
{
	int ret;
	FILE *f;
	CTFTrace &trace = CTFTrace::getInstance();

	std::string userPath = trace.getUserTracePath();
	std::string path = userPath + "/metadata";

	f = fopen(path.c_str(), "w");
	FatalErrorHandler::failIf(f == NULL, std::string("Instrumentation: ctf: writting metadata file: ") + strerror(errno));

	fputs(meta_header, f);
	fputs(meta_typedefs, f);
	fputs(meta_trace, f);
	fputs(meta_env, f);
	printCommonMetaEnv(f);
	fprintf(f, meta_clock, trace.getTimeCorrection());

	// print context additional structures
	for (auto it = contexes.begin(); it != contexes.end(); ++it) {
		CTFAPI::CTFContext *context = (*it);
		fputs(context->getDataStructuresMetadata(), f);
	}

	fprintf(f, meta_streamBounded,   CTFStreamBoundedId);
	fprintf(f, meta_streamUnbounded, CTFStreamUnboundedId);

	// print events bound to first stream
	for (auto it = events.begin(); it != events.end(); ++it) {
		CTFAPI::CTFEvent *event = it->second;
		writeEventMetadata(f, event, CTFStreamBoundedId);
	}

	// Print event bound to the second stream.
	//
	// Due to a CTF language limitation where each event definition can only
	// belong to a single ctf stream, we must copy each event definition
	// twice, one for each ctf stream
	for (auto it = events.begin(); it != events.end(); ++it) {
		CTFAPI::CTFEvent *event = it->second;
		writeEventMetadata(f, event, CTFStreamUnboundedId);
	}

	ret = fclose(f);
	FatalErrorHandler::failIf(ret, std::string("Instrumentation: ctf: closing metadata file: ") + strerror(errno));
}


static std::string trim(
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

bool CTFAPI::CTFUserMetadata::loadEnabledEvents(const char *file)
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

		_enabledEvents.push_back(trimmedLine);
	}

	streamFile.close();

	FatalErrorHandler::warnIf(
		_enabledEvents.size() == 0,
		"CTF: Kernel: Kernel tracepoints file found, but no event enabled."
	);

	return _enabledEvents.size() > 0;
}


void CTFAPI::CTFUserMetadata::refineEvents()
{
	const char *defaultRuntimeTracepointsFile = "./nanos6_runtime_tracepoints.txt";
	if (!loadEnabledEvents(defaultRuntimeTracepointsFile))
		return;

	// Disable all events
	for (auto it = events.begin(); it != events.end(); ++it) {
		CTFAPI::CTFEvent *event = it->second;
		event->disable();
	}

	// enable user-selected events
	for (std::string eventName : _enabledEvents) {
		CTFEvent *event = events[eventName];
		event->enable();
	}

	// print enabled events
	for (auto it = events.begin(); it != events.end(); ++it) {
		CTFAPI::CTFEvent *event = it->second;
		if (event->isEnabled()) {
			std::cout << event->getName() << std::endl;
		}
	}
}
