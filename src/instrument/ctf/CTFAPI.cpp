/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <fstream>
#include <string>
#include <iostream>
#include <cassert>
#include <errno.h>

#include <lowlevel/FatalErrorHandler.hpp>

#include "CTFAPI.hpp"

#define xstr(s) str(s)
#define str(s) #s

static const char *kernelMetadata = "/* CTF 1.8 */\n"
	"\n"
	"typealias integer { size = 8; align = 8; signed = false; }  := uint8_t;\n"
	"typealias integer { size = 16; align = 8; signed = false; } := uint16_t;\n"
	"typealias integer { size = 32; align = 8; signed = false; } := uint32_t;\n"
	"typealias integer { size = 64; align = 8; signed = false; } := uint64_t;\n"
	"typealias integer { size = 64; align = 8; signed = false; } := unsigned long;\n"
	"typealias integer { size = 5; align = 1; signed = false; }  := uint5_t;\n"
	"typealias integer { size = 27; align = 1; signed = false; } := uint27_t;\n"
	"\n"
	"trace {\n"
	"	major = 1;\n"
	"	minor = 8;\n"
	"	byte_order = le;\n"
	"	packet.header := struct {\n"
	"		uint32_t magic;\n"
	"		uint32_t stream_id;\n"
	"	};\n"
	"};\n"
	"\n"
	"env {\n"
	"	domain = \"kernel\";\n"
	"	tracer_name = \"lttng-modules\";\n"
	"	tracer_major = 2;\n"
	"	tracer_minor = 11;\n"
	"	tracer_patchlevel = 0;\n"
	"};\n"
	"\n"
	"clock {\n"
	"	name = \"monotonic\";\n"
	"	description = \"Monotonic Clock\";\n"
	"	freq = 1000000000; /* Frequency, in Hz */\n"
	"	/* clock value offset from Epoch is: offset * (1/freq) */\n"
	"	offset = 1578378831114078890;\n"
	"};\n"
	"\n"
	"typealias integer {\n"
	"	size = 64;\n"
	"	align = 8;\n"
	"	signed = false;\n"
	"	map = clock.monotonic.value;\n"
	"} := uint64_clock_monotonic_t;\n"
	"\n"
	"stream {\n"
	"	id = 0;\n"
	"	packet.context := struct {\n"
	"		uint32_t cpu_id;\n"
	"	};\n"
	"	event.header := struct {\n"
	"		uint32_t id;\n"
	"		uint64_clock_monotonic_t timestamp;\n"
	"	};\n"
	"};\n"
	"\n"
	"event {\n"
	"	name = \"sched_switch\";\n"
	"	id = 0;\n"
	"	stream_id = 0;\n"
	"	fields := struct {\n"
	"		integer { size = 8; align = 8; signed = 0; encoding = UTF8; base = 10; } _prev_comm[16];\n"
	"		integer { size = 32; align = 8; signed = 1; encoding = none; base = 10; } _prev_tid;\n"
	"		integer { size = 32; align = 8; signed = 1; encoding = none; base = 10; } _prev_prio;\n"
	"		integer { size = 64; align = 8; signed = 1; encoding = none; base = 10; } _prev_state;\n"
	"		integer { size = 8; align = 8; signed = 0; encoding = UTF8; base = 10; } _next_comm[16];\n"
	"		integer { size = 32; align = 8; signed = 1; encoding = none; base = 10; } _next_tid;\n"
	"		integer { size = 32; align = 8; signed = 1; encoding = none; base = 10; } _next_prio;\n"
	"	};\n"
	"};\n"
	"\n"
	"event {\n"
	"	name = \"syscall_entry_futex\";\n"
	"	id = 188;\n"
	"	stream_id = 0;\n"
	"	fields := struct {\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _uaddr;\n"
	"		integer { size = 32; align = 8; signed = 1; encoding = none; base = 10; } _op;\n"
	"		integer { size = 32; align = 8; signed = 0; encoding = none; base = 10; } _val;\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _utime;\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _uaddr2;\n"
	"		integer { size = 32; align = 8; signed = 0; encoding = none; base = 10; } _val3;\n"
	"	};\n"
	"};\n"
	"\n"
	"event {\n"
	"	name = \"syscall_exit_futex\";\n"
	"	id = 477;\n"
	"	stream_id = 0;\n"
	"	fields := struct {\n"
	"		integer { size = 64; align = 8; signed = 1; encoding = none; base = 10; } _ret;\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _uaddr;\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _uaddr2;\n"
	"	};\n"
	"};\n"
	"\n"
	"event {\n"
	"	name = \"syscall_entry_mmap\";\n"
	"	id = 13;\n"
	"	stream_id = 0;\n"
	"	fields := struct {\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 16; } _addr;\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _len;\n"
	"		integer { size = 32; align = 8; signed = 1; encoding = none; base = 10; } _prot;\n"
	"		integer { size = 32; align = 8; signed = 1; encoding = none; base = 10; } _flags;\n"
	"		integer { size = 32; align = 8; signed = 1; encoding = none; base = 10; } _fd;\n"
	"		integer { size = 64; align = 8; signed = 1; encoding = none; base = 10; } _offset;\n"
	"	};\n"
	"};\n"
	"\n"
	"event {\n"
	"	name = \"syscall_exit_mmap\";\n"
	"	id = 302;\n"
	"	stream_id = 0;\n"
	"	fields := struct {\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 16; } _ret;\n"
	"	};\n"
	"};\n"
	"\n"
	"event {\n"
	"	name = \"syscall_entry_mprotect\";\n"
	"	id = 14;\n"
	"	stream_id = 0;\n"
	"	fields := struct {\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _start;\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _len;\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _prot;\n"
	"	};\n"
	"};\n"
	"\n"
	"event {\n"
	"	name = \"syscall_exit_mprotect\";\n"
	"	id = 303;\n"
	"	stream_id = 0;\n"
	"	fields := struct {\n"
	"		integer { size = 64; align = 8; signed = 1; encoding = none; base = 10; } _ret;\n"
	"	};\n"
	"};\n"
	"\n"
	"event {\n"
	"	name = \"syscall_entry_munmap\";\n"
	"	id = 15;\n"
	"	stream_id = 0;\n"
	"	fields := struct {\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _addr;\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 10; } _len;\n"
	"	};\n"
	"};\n"
	"\n"
	"event {\n"
	"	name = \"syscall_exit_munmap\";\n"
	"	id = 304;\n"
	"	stream_id = 0;\n"
	"	fields := struct {\n"
	"		integer { size = 64; align = 8; signed = 1; encoding = none; base = 10; } _ret;\n"
	"	};\n"
	"};\n"
	"\n";


static const char *userMetadata = "/* CTF 1.8 */\n"
	"\n"
	"typealias integer { size = 8; align = 8; signed = false; }  := uint8_t;\n"
	"typealias integer { size = 16; align = 8; signed = false; } := uint16_t;\n"
	"typealias integer { size = 32; align = 8; signed = false; } := uint32_t;\n"
	"typealias integer { size = 64; align = 8; signed = false; } := uint64_t;\n"
	"typealias integer { size = 64; align = 8; signed = false; } := unsigned long;\n"
	"typealias integer { size = 5; align = 1; signed = false; }  := uint5_t;\n"
	"typealias integer { size = 27; align = 1; signed = false; } := uint27_t;\n"
	"\n"
	"trace {\n"
	"	major = 1;\n"
	"	minor = 8;\n"
	"	byte_order = le;\n"
	"	packet.header := struct {\n"
	"		uint32_t magic;\n"
	"		uint32_t stream_id;\n"
	"	};\n"
	"};\n"
	"\n"
	"env {\n"
	"	domain = \"ust\";\n"
	"	tracer_name = \"lttng-ust\";\n"
	"	tracer_major = 2;\n"
	"	tracer_minor = 11;\n"
	"	tracer_patchlevel = 0;\n"
	"};\n"
	"\n"
	"clock {\n"
	"	name = \"monotonic\";\n"
	"	description = \"Monotonic Clock\";\n"
	"	freq = 1000000000; /* Frequency, in Hz */\n"
	"	/* clock value offset from Epoch is: offset * (1/freq) */\n"
	"	offset = 1578378831114078890;\n"
	"};\n"
	"\n"
	"typealias integer {\n"
	"	size = 64;\n"
	"	align = 8;\n"
	"	signed = false;\n"
	"	map = clock.monotonic.value;\n"
	"} := uint64_clock_monotonic_t;\n"
	"\n"
	"stream {\n"
	"	id = 0;\n"
	"	packet.context := struct {\n"
	"		uint32_t cpu_id;\n"
	"	};\n"
	"	event.header := struct {\n"
	"		uint8_t id;\n"
	"		uint64_clock_monotonic_t timestamp;\n"
	"	};\n"
	"};\n"
	"\n"
	"event {\n"
	"	name = \"nanos6:task_execute\";\n"
	"	id = " xstr(TP_NANOS6_TASK_EXECUTE) ";\n"
	"	stream_id = 0;\n"
	"	fields := struct {\n"
	"		integer { size = 64; align = 8; signed = 0; encoding = none; base = 16; } _addr;\n"
	"		integer { size = 32; align = 8; signed = 0; encoding = none; base = 10; } _id;\n"
	"	};\n"
	"};\n"
	"\n"
	"event {\n"
	"	name = \"nanos6:task_end\";\n"
	"	id = " xstr(TP_NANOS6_TASK_END) ";\n"
	"	stream_id = 0;\n"
	"	fields := struct {\n"
	"		integer { size = 32; align = 8; signed = 0; encoding = none; base = 10; } _id;\n"
	"	};\n"
	"};\n"
	"\n"
	"event {\n"
	"	name = \"nanos6:task_block\";\n"
	"	id = " xstr(TP_NANOS6_TASK_BLOCK) ";\n"
	"	stream_id = 0;\n"
	"	fields := struct {\n"
	"		integer { size = 32; align = 8; signed = 0; encoding = none; base = 10; } _id;\n"
	"	};\n"
	"};\n"
	"\n";

static int mk_packet_header(char *buf, uint64_t *head)
{
	struct __attribute__((__packed__)) packet_header {
		uint32_t magic;
		uint32_t stream_id;
	};

	const int pks = sizeof(struct packet_header);
	struct packet_header *pk;

	pk = (struct packet_header *) &buf[*head];
	*pk = (struct packet_header) {
		.magic = 0xc1fc1fc1,
		.stream_id = 0
	};

	*head += pks;

	return 0;
}

static int mk_packet_context(char *buf, size_t *head, uint32_t cpu_id)
{
	struct __attribute__((__packed__)) packet_context {
		uint32_t cpu_id;
	};

	const int pks = sizeof(struct packet_context);
	struct packet_context *pk;

	pk = (struct packet_context *) &buf[*head];
	*pk = (struct packet_context) {
		.cpu_id = cpu_id,
	};

	*head += pks;

	return 0;
}

void CTFAPI::greetings(void)
{
	std::cout << "!!!!!!!!!!!!!!!!CTF API UP & Running!!!!!!!!!!!!!!!!" << std::endl;
}

void CTFAPI::writeUserMetadata(std::string directory)
{
	std::ofstream out(directory + "/metadata");
	out << std::string(userMetadata);
	out.close();
}

void CTFAPI::writeKernelMetadata(std::string directory)
{
	std::ofstream out(directory + "/metadata");
	out << std::string(kernelMetadata);
	out.close();
}

void CTFAPI::addStreamHeader(Instrument::CTFStream &stream)
{
	// we don't need to mask the head because the buffer is at least 1 page
	// long and at this point it's empty
	mk_packet_header (stream.buffer, &stream.head);
	mk_packet_context(stream.buffer, &stream.head, stream.cpuId);
}
