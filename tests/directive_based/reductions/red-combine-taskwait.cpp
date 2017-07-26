/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

/*
 * Test whether a taskwait will force reduction combination
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

int main()
{
	long activeCPUs = nanos_get_num_cpus();
	double delayMultiplier = sqrt(activeCPUs);
	
	int x = 0;
	int sync = 0;
	
	tap.registerNewTests(2);
	tap.begin();
	
	for (int i = 0; i < activeCPUs*4 - 1; ++i) {
		#pragma oss task reduction(+: x) in(sync)
		{
			int id = ++numTasks;
			tap.emitDiagnostic("Task ", id, "/", activeCPUs*4,
				" (REDUCTION) is executed");
			
			x++;
		}
	}
	
	#pragma oss task reduction(+: x) out(sync)
	{
		int id = ++numTasks;
		tap.emitDiagnostic("Task ", id, "/", activeCPUs*4,
			" (REDUCTION) is executed");
		
		ready = true;
		
		x++;
		sync = 0;
	}
	
	// Wait for tasks to finish
	tap.timedEvaluate(
			[&]() {return (numTasks.load() == activeCPUs*4) && ready.load();},
			SUSTAIN_MICROSECONDS*delayMultiplier,
			"All previous reduction tasks have been executed",
			/* weak */ true);
		
	// Taskwait combines the reduction
	#pragma oss taskwait
	
	std::ostringstream oss;
	oss << "Expected reduction computation when taskwait is reached";
	tap.evaluate(x == activeCPUs*4, oss.str());
	
	tap.end();
}
