/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/


#include <errno.h>
#include <linux/version.h>
#include <sys/utsname.h>

#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include "CTFTrace.hpp"
#include "CTFKernelMetadata.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "stream/CTFStream.hpp"
#include "stream/CTFKernelStream.hpp"
#include "support/JsonFile.hpp"


// TODO factor out common operations/data with CTFMetadata
ConfigVariableList<std::string> CTFAPI::CTFKernelMetadata::_kernelEventPresets("instrument.ctf.events.kernel.presets", {});
ConfigVariable<std::string> CTFAPI::CTFKernelMetadata::_kernelEventFile("instrument.ctf.events.kernel.file");
ConfigVariableList<std::string> CTFAPI::CTFKernelMetadata::_kernelExcludedEvents("instrument.ctf.events.kernel.exclude", {});

// TODO manage old alternative event names
std::map<std::string, std::vector<std::string> > CTFAPI::CTFKernelMetadata::_kernelEventPresetMap {
	{"context_switch",
		{
			"sched_switch"
		}
	},
	{"preemption",
		{
			"irq_handler_entry",
			"irq_handler_exit",
			"softirq_entry",
			"softirq_exit",
			"sched_switch"
		}
	},
	{"syscall",
		{
			"sched_switch",
			"__syscalls"
		}
	}
};


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
	"	cpu_list = \"%s\";\n"
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

bool CTFAPI::CTFKernelMetadata::getSystemInformation()
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

	uint64_t minimumKernelVersion = CTFKernelStream::minimumKernelVersion();

	return (LINUX_VERSION_CODE >= minimumKernelVersion);
}

void CTFAPI::CTFKernelMetadata::loadKernelDefsFile(const char *file)
{
	JsonFile configFile = JsonFile(file);

	if (configFile.fileExists()) {
		configFile.loadData();

		// Navigate through the file and extract tracepoint definitions
		configFile.getRootNode()->traverseChildrenNodes(
			[&](const std::string &name, const JsonNode<> &node) {
				if (name == "meta") {
					// The "meta" node is allways found first
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

					_kernelEventMap.emplace(name, std::make_pair(id, format));
					assert(_eventSizes.size() > id);
					_eventSizes[id] = size;
				}
			}
		);

		bool success = (_maxEventId != (ctf_kernel_event_id_t) -1) && (_kernelEventMap.size() > 0);
		FatalErrorHandler::failIf(!success, "CTF: kernel: Kernel events definitions file present, but corrupted");
	} else {
		FatalErrorHandler::fail("CTF: kernel: Kernel events definitions file not found");
	}
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

bool CTFAPI::CTFKernelMetadata::addEventsInPresets()
{
	if (!_kernelEventPresets.isPresent())
		return false;

	size_t enabledEventsBefore = _enabledEventNames.size();
	for (std::string preset : _kernelEventPresets) {
		try {
			for (std::string event : _kernelEventPresetMap.at(preset)) {
				if (event == "__syscalls") {
					_enableSyscalls = true;
					continue;
				}
				_enabledEventNames.insert(event);
			}
		} catch(std::out_of_range &e) {
			FatalErrorHandler::fail("CTF: Kernel: The preset \"", preset, "\" is not valid");
		}
	}
	size_t enabledEventsAfter = _enabledEventNames.size();

	return _enableSyscalls || (enabledEventsBefore != enabledEventsAfter);
}

bool CTFAPI::CTFKernelMetadata::addEventsInFile()
{
	std::string line;
	std::ifstream streamFile;
	size_t enabledEventsAfter, enabledEventsBefore;

	if (!_kernelEventFile.isPresent())
		return false;

	std::string file = _kernelEventFile.getValue().c_str();

	streamFile.open(file.c_str());
	if (!streamFile.is_open()) {
		FatalErrorHandler::fail(
			"CTF: Kernel: Kernel tracepoints file provided but it could not be opened: ",
			strerror(errno)
		);
	}

	enabledEventsBefore = _enabledEventNames.size();

	while (std::getline(streamFile, line)) {
		std::string trimmedLine = trim(line);
		if (trimmedLine[0] == '#' || trimmedLine == "")
			continue;

		_enabledEventNames.insert(trimmedLine);
	}

	streamFile.close();

	enabledEventsAfter = _enabledEventNames.size();
	FatalErrorHandler::warnIf(
		enabledEventsBefore == enabledEventsAfter,
		"CTF: Kernel: Kernel tracepoints file found, but no event enabled."
	);

	return enabledEventsBefore != enabledEventsAfter;
}

void CTFAPI::CTFKernelMetadata::addPatternDependentEvents()
{
	// Enable all events starting with "sys_" (all syscall events) but the
	// generic "sys_enter" and "sys_exit". We prefer enabling syscall
	// specific events e.g. sys_enter_open/sys_exit_open rather than the
	// generic ones sys_enter/sys_exit because generic events always have
	// six arguemnts, even if the specific syscall tracepoint has less.
	if (_enableSyscalls) {
		kernel_event_map_t::iterator it;
		for (it = _kernelEventMap.begin(); it != _kernelEventMap.end(); it++) {
			if (it->first.rfind("sys_", 0) == 0) {
				if (it->first == "sys_enter" || it->first == "sys_exit")
					continue;
				_enabledEventNames.insert(it->first);
			}
		}
	}
}

void CTFAPI::CTFKernelMetadata::excludeEvents()
{
	if (_kernelExcludedEvents.isPresent()) {
		for (std::string event : _kernelExcludedEvents) {
			_enabledEventNames.erase(event);
		}
	}
}

void CTFAPI::CTFKernelMetadata::translateEvents()
{
	for (std::string eventName : _enabledEventNames) {
		try {
			auto const& entry = _kernelEventMap.at(eventName);
			ctf_kernel_event_id_t id = entry.first;
			_enabledEventIds.push_back(id);
		} catch(std::out_of_range &e) {
			FatalErrorHandler::fail("CTF: Kernel: The event \"", eventName, "\" is not found into the kernel tracepoint definition file");
		}
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
		_cpuList.c_str(),
		trace.getBinaryName(),
		trace.getPid());
	fprintf(f, meta_clock, trace.getAbsoluteStartTimestamp());
	fprintf(f, meta_stream, CTFKernelStreamId);

	for (std::string event : _enabledEventNames) {
		auto const& entry = _kernelEventMap.at(event);
		fprintf(f, meta_event,
			event.c_str(),       // event name
			entry.first,         // event id
			CTFKernelStreamId,   // stream id
			entry.second.c_str() // event format
		);
	}

	ret = fclose(f);
	if (ret) {
		FatalErrorHandler::warn("CTF: Kernel: closing metadata file: ", strerror(errno));
	}
}

void CTFAPI::CTFKernelMetadata::parseKernelEventDefinitions()
{
	const char kernelDefsParser[] = "ctfkerneldefs";

	// Call an external python command to parse the kernel event definitions file
	// TODO reimplement the ctfkerneldefs tool in c++
	CTFTrace &trace = CTFTrace::getInstance();
	std::string parser = trace.searchPythonCommand(kernelDefsParser);
	std::string tracePath = trace.getTemporalTracePath();
	std::string kernelDefsFileName = tracePath + "/nanos6_kerneldefs.json";
	std::string command = parser + " --file=\"" + kernelDefsFileName + "\"";

	// Parse the kernel event definitions files and generate json
	int ret = system(command.c_str());
	FatalErrorHandler::failIf(
		ret == -1,
		"ctf: kernel: Could not parse kernel event definitions file: ",
		strerror(errno)
	);

	// Load the generated json
	loadKernelDefsFile(kernelDefsFileName.c_str());
}

void CTFAPI::CTFKernelMetadata::initialize()
{
	// Check for user requested events
	bool presets      = addEventsInPresets();
	bool eventsInFile = addEventsInFile();
	if (!presets && !eventsInFile)
		return;

	// Check system requirements
	bool sysInfo = getSystemInformation();
	if (!sysInfo) {
		FatalErrorHandler::warn(
			"CTF Kernel tracing requires Linux Kernel >= 4.1.0, but Nanos6 was compiled against an older kernel: Disabling Kernel tracing."
		);
		return;
	}

	// System requirements met and there are events to enable. Let's now
	// load the kernel events definition files. Failure in doing so will
	// halt the execution of Nanos6
	parseKernelEventDefinitions();

	// Next, load events that depend on kernel event definition files (we
	// are not adding specific events to the list, but events that match a
	// specific pattern.
	addPatternDependentEvents();

	// With the whole list of events requested by the user, now disable events in the exclude list
	excludeEvents();

	// After excluding, the event list might be empty. Disable if that's the case
	if (_enabledEventNames.size() == 0) {
		FatalErrorHandler::warn(
			"All CTF kernel events have been excluded: Disabling kernel tracing"
		);
		_enabled = false;
		return;
	}

	// Finally, translate event names to event Id's
	translateEvents();

	// Event list ready!
	_enabled = true;
}
