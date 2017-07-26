/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

/*
 * Test whether a taskwait will mark a reduction task to force combination
 *
 */

#include <atomic>

#include <math.h>
#include <unistd.h>
#include <stdio.h>

#include <nanos6/debug.h>

#include "TestAnyProtocolProducer.hpp"


#define SUSTAIN_MICROSECONDS 100000L


TestAnyProtocolProducer tap;
std::atomic_int numTasks(0);
std::atomic_bool ready(false);

auto isReady = [&]() {
	bool var = ready.load();
	return var;
};

int main()
{
	long activeCPUs = nanos_get_num_cpus();
	double delayMultiplier = sqrt(activeCPUs);
	
	int x = 0;
	int sync = 0;
	
	tap.registerNewTests(activeCPUs*4 + 1);
	tap.begin();
	
	for (int i = 0; i < activeCPUs*4; ++i) {
		#pragma oss task reduction(+: x) in(sync)
		{
			int id = ++numTasks;
			tap.emitDiagnostic("Task ", id, "/", activeCPUs*4,
				" (REDUCTION) enters synchronization");
			
			std::ostringstream oss;
			oss << "Task " << id << "/" << activeCPUs*4 <<
				" (REDUCTION) is executed after all tasks have been submitted";
			tap.timedEvaluate(
				isReady,
				SUSTAIN_MICROSECONDS*delayMultiplier,
				oss.str(),
				/* weak */ true
			);
			usleep(SUSTAIN_MICROSECONDS*delayMultiplier);
			
			x++;
		}
	}
	
	// Wake up reduction tasks
	// WARNING: This would _ideally_ be done inside the taskwait, so we add an
	// extra wait in the reduction tasks
	tap.emitDiagnostic("All tasks submitted, unblocking held tasks");
	ready = true;
	
	// Taskwait combines the reduction
	#pragma oss taskwait
	
	std::ostringstream oss;
	oss << "Expected reduction computation when taskwait is reached";
	tap.evaluateWeak(x == activeCPUs*4, oss.str(),
		/* weakDetail */ "Can only be tested using sleeps, and therefore not deterministic");
	
	tap.end();
}
