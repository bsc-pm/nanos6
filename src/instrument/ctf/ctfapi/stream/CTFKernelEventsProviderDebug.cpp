/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/


#include "../CTFTrace.hpp"
#include "CTFKernelEventsProvider.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

struct __attribute__((__packed__)) event_sched_switch {
	CTFAPI::CTFKernelEventsProvider::EventCommon common;
	char  prev_comm[16];
	pid_t prev_pid;
	int   prev_prio;
	long  prev_state;
	char  next_comm[16];
	pid_t next_pid;
	int   next_prio;
};

struct __attribute__((__packed__)) event_sched_wakeup {
	CTFAPI::CTFKernelEventsProvider::EventCommon common;
	char  comm[16];
	pid_t pid;
	int   prio;
	int   success;
	int   target_cpu;
};

#define PERF_HEADER_ENTRY(name)  \
case name: \
	output = output + (#name) + " "; \
	break;

static std::string decodePerfHeaderType(struct perf_event_header *header)
{
	std::string output;

	switch (header->type) {
	PERF_HEADER_ENTRY(PERF_RECORD_MMAP)
	PERF_HEADER_ENTRY(PERF_RECORD_LOST)
	PERF_HEADER_ENTRY(PERF_RECORD_COMM)
	PERF_HEADER_ENTRY(PERF_RECORD_EXIT)
	PERF_HEADER_ENTRY(PERF_RECORD_THROTTLE)
	PERF_HEADER_ENTRY(PERF_RECORD_UNTHROTTLE)
	PERF_HEADER_ENTRY(PERF_RECORD_FORK)
	PERF_HEADER_ENTRY(PERF_RECORD_READ)
	PERF_HEADER_ENTRY(PERF_RECORD_SAMPLE)
	default:
		output = output + "[type" + std::to_string(header->type) + " not known]";
	}

	switch (header->misc & PERF_RECORD_MISC_CPUMODE_MASK) {
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_CPUMODE_UNKNOWN)
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_KERNEL)
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_USER)
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_HYPERVISOR)
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_GUEST_KERNEL)
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_GUEST_USER)
#ifdef PERF_RECORD_MISC_MMAP_DATA
	PERF_HEADER_ENTRY(PERF_RECORD_MISC_MMAP_DATA)
#endif
	default:
		output = output + "[misc " + std::to_string(header->misc) + " not known]";
	}

	if (header->misc & PERF_RECORD_MISC_EXACT_IP) {
		output += "PERF_RECORD_MISC_EXACT_IP";
#ifdef PERF_RECORD_MISC_MMAP_DATA
	} else if (header->misc & PERF_RECORD_MISC_MMAP_DATA) {
		output += "PERF_RECORD_MISC_MMAP_DATA";
#endif
	} else if (header->misc & PERF_RECORD_MISC_EXT_RESERVED) {
		output += "PERF_RECORD_MISC_EXT_RESERVED";
	}

	return output;
}

void CTFAPI::CTFKernelEventsProvider::warnUnknownPerfRecord(struct perf_event_header *header)
{
#ifndef NDEBUG
	FatalErrorHandler::warn("Ignoring unknown perf record: ", decodePerfHeaderType(header));
#endif
}

void CTFAPI::CTFKernelEventsProvider::checkPerfRecordSampleSize(PerfRecordSample *perfRecordSample)
{
#ifndef NDEBUG
	size_t total, recordExtra;
	size_t unknown;

	total = perfRecordSample->header.size;
	recordExtra = sizeof(uint64_t) /*time*/ + sizeof(uint32_t); /*size*/

	unknown = total - sizeof(struct perf_event_header)
			- recordExtra
			- perfRecordSample->size;

	if (unknown) {
		FatalErrorHandler::warn(
			"Sample record unknown bytes: ", unknown,
			" header.size  = ", perfRecordSample->header.size,
			" size         = ", perfRecordSample->size,
			" record extra = ", recordExtra,
			" header type  = ", decodePerfHeaderType(&perfRecordSample->header)
		);
	}
#endif
}

void CTFAPI::CTFKernelEventsProvider::checkPerfEventSize(
	PerfRecordSample *perfRecordSample,
	ctf_kernel_event_size_t expectedFieldsSize
) {
#ifndef NDEBUG
	EventCommon *commonFields = (EventCommon *) perfRecordSample->data;
	ctf_kernel_event_id_t eventId = commonFields->common_type;
	size_t commonFieldsSize = sizeof(EventCommon);
	ssize_t unknown;
	unsigned char align;
	unsigned char padding;

	// data is 64 bit aligned, padded with 0
	align = ((((uintptr_t) perfRecordSample->data) + commonFieldsSize + expectedFieldsSize) & 0x7UL);
	padding = align ? 8 - align : 0;

	// The data returned in perfRecordSample->data should only contain the
	// common tracepoint fields, the expected fields for this specific
	// tracepoint, and padding. The expectedFieldsSize accounts for the both
	// common and specific tracepoint fields.
	unknown = perfRecordSample->size - commonFieldsSize - expectedFieldsSize - padding;
	if (unknown == 0)
		return;

	std::cerr << "event " << eventId << " unknown bytes: " << std::dec << unknown << std::endl << 
		" header.size             = " <<            perfRecordSample->header.size    << std::endl <<
		" sizeof(header)          = " <<            sizeof(struct perf_event_header) << std::endl <<
		" data size               = " <<            perfRecordSample->size           << std::endl <<
		" common fields size      = " <<            commonFieldsSize                 << std::endl <<
		" expected tp fields size = " <<            expectedFieldsSize               << std::endl <<
		" padding                 = " << (uint64_t) padding                          << std::endl <<
		" record addr             = " <<            perfRecordSample                 << std::endl <<
		" data addr               = " << (void *)   perfRecordSample->data           << std::endl;
#endif
}

void CTFAPI::CTFKernelEventsProvider::printKernelEvent(
	PerfRecordSample *perfRecordSample,
	ctf_kernel_event_size_t payloadSize
) {
#ifndef NDEBUG
	EventCommon *commonFields = (EventCommon *) perfRecordSample->data;
	ctf_kernel_event_id_t eventId = commonFields->common_type;

	//std::cout << "event raw dump: " << std::hex;
	//for (uint64_t i = 0; i < perfRecordSample->header.size; i++) {
	//	std::cout << ((char *)perfRecordSample)[i];
	//}
	//std::cout << std::dec << std::endl;

	std::cout << perfRecordSample->time <<
		" {common_type = "          << commonFields->common_type <<
		", common_flags = "         << commonFields->common_flags <<
		", common_preempt_count = " << commonFields->common_preempt_count <<
		", common_pid = "           << commonFields->common_pid <<
		"} ";

	if (eventId == _sched_switch_id) {
		struct event_sched_switch *tp;
		tp = (struct event_sched_switch *) commonFields;
		size_t fieldsSize = sizeof(*tp) - sizeof(*commonFields);
		assert(fieldsSize == payloadSize);
		std::cout << "{prev_comm =" << tp->prev_comm <<
			", prev_pid = "  << tp->prev_pid     <<
			", prev_prio = " << tp->prev_prio    <<
			", prev_state = "<< tp->prev_state   <<
			", next_comm = " << tp->next_comm    <<
			", next_pid = "  << tp->next_pid     <<
			", next_prio = " << tp->next_prio    <<
			"}" << std::endl;
	} else if (eventId == _sched_wakeup_id) {
		struct event_sched_wakeup *tp;
		tp = (struct event_sched_wakeup *) commonFields;
		size_t fieldsSize = sizeof(*tp) - sizeof(*commonFields);
		assert(fieldsSize == payloadSize);
		std::cout << "{comm="     << tp->comm       <<
			", pid="          << tp->pid        <<
			", prio="         << tp->prio       <<
			", success = "    << tp->success    <<
			", target_cpu = " << tp->target_cpu <<
			"}" << std::endl;
	} else {
		std::cout << "unknown definition" << std::endl;
	}
#endif
}

void CTFAPI::CTFKernelEventsProvider::initializeDebug()
{
#ifndef NDEBUG
	CTFTrace &trace = CTFTrace::getInstance();
	CTFKernelMetadata *metadata = trace.getKernelMetadata();
	_sched_switch_id = metadata->getEventIdByName("sched_switch");
	_sched_wakeup_id = metadata->getEventIdByName("sched_wakeup");
#endif
}
