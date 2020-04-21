/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <fstream>
#include <cstdint>
#include <cinttypes>

#include "CTFTrace.hpp"
#include "CTFMetadata.hpp"
#include "CTFContext.hpp"

const char *CTFAPI::CTFMetadata::meta_header = "/* CTF 1.8 */\n";

const char *CTFAPI::CTFMetadata::meta_typedefs =
	"typealias integer { size = 8; align = 8; signed = false; }  := uint8_t;\n"
	"typealias integer { size = 16; align = 8; signed = false; } := uint16_t;\n"
	"typealias integer { size = 32; align = 8; signed = false; } := uint32_t;\n"
	"typealias integer { size = 64; align = 8; signed = false; } := uint64_t;\n"
	"typealias integer { size = 64; align = 8; signed = false; } := unsigned long;\n"
	"typealias integer { size = 5; align = 1; signed = false; }  := uint5_t;\n"
	"typealias integer { size = 27; align = 1; signed = false; } := uint27_t;\n\n";

const char *CTFAPI::CTFMetadata::meta_trace =
	"trace {\n"
	"	major = 1;\n"
	"	minor = 8;\n"
	"	byte_order = le;\n"
	"	packet.header := struct {\n"
	"		uint32_t magic;\n"
	"		uint32_t stream_id;\n"
	"	};\n"
	"};\n\n";

const char *CTFAPI::CTFMetadata::meta_env =
	"env {\n"
	"	/* Trace Compass variables */\n"
	"	domain = \"ust\";\n"
	"	tracer_name = \"lttng-ust\";\n"
	"	tracer_major = 2;\n"
	"	tracer_minor = 11;\n"
	"	tracer_patchlevel = 0;\n"
	"	/* ctf2prv converter variables */\n"
	"	ncpus = %" PRIu16 ";\n"
	"};\n\n";

const char *CTFAPI::CTFMetadata::meta_clock =
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

const char *CTFAPI::CTFMetadata::meta_streamBounded =
	"stream {\n"
	"	id = 0;\n"
	"	packet.context := struct {\n"
	"		uint16_t cpu_id;\n"
	"	};\n"
	"	event.header := struct {\n"
	"		uint8_t id;\n"
	"		uint64_clock_monotonic_t timestamp;\n"
	"	};\n"
	"};\n\n";

const char *CTFAPI::CTFMetadata::meta_streamUnbounded =
	"stream {\n"
	"	id = 1;\n"
	"	packet.context := struct {\n"
	"		uint16_t cpu_id;\n"
	"	};\n"
	"	event.header := struct {\n"
	"		uint8_t id;\n"
	"		uint64_clock_monotonic_t timestamp;\n"
	"	};\n"
	"	event.context := struct {\n"
	"		uint16_t event_cpu_id;\n"
	"	};\n"
	"};\n\n";

const char *CTFAPI::CTFMetadata::meta_eventMetadataId =
	"event {\n"
	"	name = \"%s\";\n"
	"	id = %d;\n";

const char *CTFAPI::CTFMetadata::meta_eventMetadataStreamId =
	"	stream_id = %d;\n";

const char *CTFAPI::CTFMetadata::meta_eventMetadataFields =
	"	fields := struct {\n"
	"%s"
	"	};\n"
	"};\n\n";

void CTFAPI::CTFMetadata::writeEventContextMetadata(FILE *f, CTFAPI::CTFEvent *event)
{
	std::vector<CTFAPI::CTFContext *> &contexes = event->getContexes();

	if (contexes.empty())
		return;

	fprintf(f, "	context := {\n");

	for (auto it = contexes.begin(); it != contexes.end(); ++it) {
		CTFAPI::CTFContext *context = (*it);
		fputs(context->getMetadata(), f);
	}

	fprintf(f, "	}\n");

}

void CTFAPI::CTFMetadata::writeEventMetadata(FILE *f, CTFAPI::CTFEvent *event, int streamId)
{
	fprintf(f, meta_eventMetadataId, event->getName(), event->getEventId());
	fprintf(f, meta_eventMetadataStreamId, streamId);
	writeEventContextMetadata(f, event);
	fprintf(f, meta_eventMetadataFields, event->getMetadataFields());
}

void CTFAPI::CTFMetadata::writeMetadataFile(std::string userPath)
{
	int ret;
	FILE *f;
	std::string path;
	CTFTrace &trace = CTFTrace::getInstance();

	path = userPath + "/metadata";

	f = fopen(path.c_str(), "w");
	FatalErrorHandler::failIf(f == NULL, std::string("Instrumentation: ctf: writting metadata file: ") + strerror(errno));

	fputs(meta_header, f);
	fputs(meta_typedefs, f);
	fputs(meta_trace, f);
	fprintf(f, meta_env, totalCPUs);
	fprintf(f, meta_clock, trace.getAbsoluteStartTimestamp());
	fputs(meta_streamBounded, f);
	fputs(meta_streamUnbounded, f);

	// print events bound to first stream
	for (auto it = events.begin(); it != events.end(); ++it) {
		CTFAPI::CTFEvent *event = (*it);
		writeEventMetadata(f, event, 0);
	}

	//// TODO print events bound to second stream
	//for (auto it = events.begin(); it != events.end(); ++it) {
	//	CTFAPI::CTFEvent *event = (*it);
	//	writeEventMetadata(f, event, 1);
	//}

	ret = fclose(f);
	FatalErrorHandler::failIf(ret, std::string("Instrumentation: ctf: closing metadata file: ") + strerror(errno));
}

