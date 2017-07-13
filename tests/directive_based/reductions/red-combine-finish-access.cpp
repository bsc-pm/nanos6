/*
 * Test whether removing a reduction dataccess with a posterior
 * non-reduction access triggers the combination of the reduction
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
