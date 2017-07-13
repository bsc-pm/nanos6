/*
 * Test whether adding a new dataccess combines pending reduction
 *
 * This test only works with more than a CPU
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
	
	for (int i = 0; i < activeCPUs*4; ++i) {
		#pragma oss task reduction(+: x) in(sync)
		{
			int id = ++numTasks;
			tap.emitDiagnostic("Task ", id, "/", activeCPUs*4,
				" (REDUCTION) is executed");
			
			x++;
		}
	}
	
	#pragma oss task out(sync)
	{
		ready = true;
		
		sync = 0;
	}
	
	// Wait for tasks to finish
	tap.timedEvaluate(
			[&]() {return (numTasks.load() == activeCPUs*4) && ready.load();},
			SUSTAIN_MICROSECONDS*delayMultiplier,
			"All previous reduction tasks have been executed",
			/* weak */ true);
		
	// New access that combines the reduction
	#pragma oss task in(x)
	{
		std::ostringstream oss;
		oss << "Expected reduction computation when task " << activeCPUs*4 + 1 <<
			" (READ) is executed";
		tap.evaluate(x == activeCPUs*4, oss.str());
	}
	
	#pragma oss taskwait
	
	tap.end();
}
