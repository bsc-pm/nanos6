/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "RuntimeInfoEssentials.hpp"
#include "system/RuntimeInfo.hpp"

#include "version/VersionInfo.hpp"

#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>


static inline void addResourceLimitReportEntry(int resource, std::string const &name, std::string const &description, std::string const &units = "")
{
	struct rlimit rlim;
	
	int rc = getrlimit(resource, &rlim);
	if (rc == -1) {
		char *error = strerror(errno);
		RuntimeInfo::addEntry(name, description, error);
	} else if (rlim.rlim_cur == RLIM_INFINITY) {
		RuntimeInfo::addEntry(name, description, "unlimited", units);
	} else {
		RuntimeInfo::addEntry(name, description, rlim.rlim_cur, units);
	}
}


void RuntimeInfoEssentials::initialize()
{
	RuntimeInfo::addEntry("version", "Runtime Version", nanos6_version);
	RuntimeInfo::addEntry("branch", "Runtime Branch", nanos6_branch);
	RuntimeInfo::addEntry("compiler_version", "Runtime Compiler Version", nanos6_compiler_version);
	RuntimeInfo::addEntry("compiler_flags", "Runtime Compiler Flags", nanos6_compiler_flags);
	
	// The following entries are generated from src/system/emit-rlimit-code.sh
#ifdef RLIMIT_CPU
	addResourceLimitReportEntry(RLIMIT_CPU, "rlimit_cpu", "Per-process CPU limit, in seconds", "seconds");
#endif
#ifdef RLIMIT_FSIZE
	addResourceLimitReportEntry(RLIMIT_FSIZE, "rlimit_fsize", "Largest file that can be created, in bytes", "bytes");
#endif
#ifdef RLIMIT_DATA
	addResourceLimitReportEntry(RLIMIT_DATA, "rlimit_data", "Maximum size of data segment, in bytes", "bytes");
#endif
#ifdef RLIMIT_STACK
	addResourceLimitReportEntry(RLIMIT_STACK, "rlimit_stack", "Maximum size of stack segment, in bytes", "bytes");
#endif
#ifdef RLIMIT_CORE
	addResourceLimitReportEntry(RLIMIT_CORE, "rlimit_core", "Largest core file that can be created, in bytes", "bytes");
#endif
#ifdef RLIMIT_RSS
	addResourceLimitReportEntry(RLIMIT_RSS, "rlimit_rss", "Largest resident set size, in bytes. This affects swapping; processes that are exceeding their resident set size will be more likely to have physical memory taken from them", "bytes");
#endif
#ifdef RLIMIT_NOFILE
	addResourceLimitReportEntry(RLIMIT_NOFILE, "rlimit_nofile", "Number of open files", "");
#endif
#ifdef RLIMIT_AS
	addResourceLimitReportEntry(RLIMIT_AS, "rlimit_as", "Address space limit", "");
#endif
#ifdef RLIMIT_NPROC
	addResourceLimitReportEntry(RLIMIT_NPROC, "rlimit_nproc", "Number of processes", "processes");
#endif
#ifdef RLIMIT_MEMLOCK
	addResourceLimitReportEntry(RLIMIT_MEMLOCK, "rlimit_memlock", "Locked-in-memory address space", "");
#endif
#ifdef RLIMIT_LOCKS
	addResourceLimitReportEntry(RLIMIT_LOCKS, "rlimit_locks", "Maximum number of file locks", "file locks");
#endif
#ifdef RLIMIT_SIGPENDING
	addResourceLimitReportEntry(RLIMIT_SIGPENDING, "rlimit_sigpending", "Maximum number of pending signals", "signals");
#endif
#ifdef RLIMIT_MSGQUEUE
	addResourceLimitReportEntry(RLIMIT_MSGQUEUE, "rlimit_msgqueue", "Maximum bytes in POSIX message queues", "bytes");
#endif
#ifdef RLIMIT_NICE
	addResourceLimitReportEntry(RLIMIT_NICE, "rlimit_nice", "Maximum nice priority allowed to raise to. Nice levels 19 .. -20 correspond to 0 .. 39 values of this resource limit", "");
#endif
#ifdef RLIMIT_RTPRIO
	addResourceLimitReportEntry(RLIMIT_RTPRIO, "rlimit_rtprio", "Maximum realtime priority allowed for non-priviledged processes", "");
#endif
#ifdef RLIMIT_RTTIME
	addResourceLimitReportEntry(RLIMIT_RTTIME, "rlimit_rttime", "Maximum CPU time in µs that a process scheduled under a real-time scheduling policy may consume without making a blocking system call before being forcibly descheduled", "µs");
#endif
}

