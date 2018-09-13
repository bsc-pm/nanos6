/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

/*
 * Test whether removing a reduction dataccess with a posterior
 * non-reduction access triggers the combination of the reduction
 *
 */

#include <math.h>
#include <unistd.h>
#include <stdio.h>

#include <nanos6/debug.h>

#include <Atomic.hpp>
#include <Functors.hpp>
#include "TestAnyProtocolProducer.hpp"


#define SUSTAIN_MICROSECONDS 100000L


using namespace Functors;


TestAnyProtocolProducer tap;
Atomic<int> numTasks(0);
Atomic<bool> ready(false);

True< Atomic<bool> > isReady(ready);


int main()
{
	long activeCPUs = nanos6_get_num_cpus();
	double delayMultiplier = sqrt(activeCPUs);
	
	int x = 0;
	
	tap.registerNewTests(activeCPUs*4 + 1);
	tap.begin();
	
	for (int i = 0; i < activeCPUs*4; ++i) {
		#pragma oss task reduction(+: x)
		{
			x++;
			
			// Hold reduction tasks until a posterior (non-reduction) task is enqueued
			
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
		}
	}
	
	#pragma oss task in(x)
	{
		std::ostringstream oss;
		oss << "Expected reduction computation when task " << activeCPUs*4 + 1 <<
			" (READ) is executed";
		tap.evaluate(x == activeCPUs*4, oss.str());
	}
	
	// Wake up reduction task now that the in task is submitted
	tap.emitDiagnostic("All tasks submitted, unblocking held tasks");
	ready = true;
	
	#pragma oss taskwait
	
	tap.end();
}
