/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>

#include <cstdlib>       /* getenv, system */
#include <cstring>       /* strcmp */
#include <fcntl.h>       /* O_WRONLY, O_CREAT */
#include <iostream>
#include <sstream>       /* stringstream */
#include <sys/stat.h>
#include <sys/sysinfo.h> /* get_nprocs */
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>


#include "TestAnyProtocolProducer.hpp"


void wrongExecution(const char *error)
{
	TestAnyProtocolProducer tap;
	tap.registerNewTests(1);
	tap.begin();
	tap.success(error);
	tap.end();
}

int main(int argc, char **argv) {
	// Make sure DLB is enabled
	char *dlbEnabled = std::getenv("NANOS6_ENABLE_DLB");
	if (dlbEnabled == 0) {
		wrongExecution("DLB is disabled, skipping this test");
		return 0;
	} else if (strcmp(dlbEnabled, "1") != 0) {
		wrongExecution("DLB is disabled, skipping this test");
		return 0;
	}

	// Check that we're using all the CPUs in the system
	long activeCPUs = get_nprocs();
	unsigned long activeCPUsNanos6 = nanos6_get_num_cpus();
	if (activeCPUs != activeCPUsNanos6) {
		wrongExecution("Can only execute this test using all available cores, skipping this test");
		return 0;
	}

	// Make sure we're using an even number of CPUs
	if (activeCPUs % 2) {
		activeCPUs -= 1;
	}

	// Make sure we're using enough CPUs
	if (activeCPUs < 4) {
		wrongExecution("Can only execute this test with 4 or more CPUs, skipping this test");
		return 0;
	}

	// Prepare the binaries of the processes to launch
	const std::string parentTestName("dlb-cpu-sharing");
	const std::string activeTestName("dlb-cpu-sharing-active-process");
	const std::string passiveTestName("dlb-cpu-sharing-passive-process");
	std::string activeCommand(argv[0]);
	std::string passiveCommand(argv[0]);

	// Construct the launch commands of the processes to launch
	const int index = activeCommand.rfind(parentTestName);
	if (index == std::string::npos) {
		wrongExecution("This test is renamed and does not match with the expected name");
		return 0;
	}
	activeCommand.replace(index, parentTestName.size(), activeTestName);
	passiveCommand.replace(index, parentTestName.size(), passiveTestName);

	long firstPassiveCPU = 0;
	long lastPassiveCPU = (activeCPUs / 2) - 1;
	long firstActiveCPU = (activeCPUs / 2);
	long lastActiveCPU = activeCPUs - 1;

	// Delete the shared memory so the subprocesses can be executed
	if (!std::system("dlb_shm -d > /dev/null 2>&1")) {
		if (fork() == 0) {
			// Build the mask of CPUs for the passive process
			cpu_set_t cpuAffinity;
			CPU_ZERO(&cpuAffinity);
			for (long id = firstPassiveCPU; id <= lastPassiveCPU; ++id) {
				CPU_SET(id, &cpuAffinity);
			}
			sched_setaffinity(0, sizeof(cpuAffinity), &cpuAffinity);

			// Launch the passive process using execv, to avoid using the
			// duplicate execution (WorkerThread, underlying runtime, ...)
			// NOTE: After the 'execv', the current process will ultimately
			// cease to exist, and switch to executing the new process with
			// a clean environment
			char *passiveProcess = new char[passiveCommand.length() + 1];
			strcpy(passiveProcess, passiveCommand.c_str());
			char *const passiveProcessArgs[] = {passiveProcess, "nanos6-testing", NULL};

			// Redirect the current output to /dev/null prior to executing 'execv'
			int fd = open("/dev/null", O_WRONLY | O_CREAT, 0666);
			int ret = dup2(fd, 1);
			if (ret == -1) {
				wrongExecution("Could not redirect output");
				return 0;
			}

			// Switch to executing the passive process
			execv(passiveProcess, passiveProcessArgs);

			// If this point is reached, something wrong happened
			wrongExecution("Could not execute a new process into the forked one");
			return 0;
		} else {
			// Build the mask of CPUs for the active process
			cpu_set_t cpuAffinity;
			CPU_ZERO(&cpuAffinity);
			for (long id = firstActiveCPU; id <= lastActiveCPU; ++id) {
				CPU_SET(id, &cpuAffinity);
			}
			sched_setaffinity(0, sizeof(cpuAffinity), &cpuAffinity);

			// No need to use execv, we will use the 'system' call so that we
			// can later wait for all children processes to end
			std::stringstream activeProcess;
			activeProcess << activeCommand << " nanos6-testing";
			std::system(activeProcess.str().c_str());

			// Wait for all children processes to finish executing
			wait(NULL);
			return 0;
		}
	} else {
		// DLB is not found, abort and skip this test
		wrongExecution("DLB not found in $PATH, skipping this test");
		return 0;
	}
}
