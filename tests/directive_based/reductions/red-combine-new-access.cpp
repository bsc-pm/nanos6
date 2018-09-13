/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

/*
 * Test whether adding a new dataccess combines pending reduction
 *
 * This test only works with more than a CPU
 *
 */

#include <math.h>
#include <unistd.h>
#include <stdio.h>

#include <nanos6/debug.h>

#include <Atomic.hpp>
#include <Functors.hpp>
#include "TestAnyProtocolProducer.hpp"


using namespace Functors;


#define SUSTAIN_MICROSECONDS 100000L


TestAnyProtocolProducer tap;
Atomic<int> numTasks(0);
Atomic<bool> ready(false);

int main()
{
	long activeCPUs = nanos6_get_num_cpus();
	double delayMultiplier = sqrt(activeCPUs);
	
	int fourTimesActiveCPUs = 4*activeCPUs;
	
	int x = 0;
	int sync = 0;
	
	tap.registerNewTests(2);
	tap.begin();
	
	for (int i = 0; i < fourTimesActiveCPUs; ++i) {
		#pragma oss task reduction(+: x) in(sync)
		{
			int id = ++numTasks;
			tap.emitDiagnostic("Task ", id, "/", fourTimesActiveCPUs,
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
	typedef Equal< Atomic<int>, int > functor1_t;
	functor1_t functor1(numTasks, fourTimesActiveCPUs);
	tap.timedEvaluate(
		And< functor1_t, Atomic<bool> > (
			functor1,
			ready
		),
		SUSTAIN_MICROSECONDS*delayMultiplier,
		"All previous reduction tasks have been executed",
		/* weak */ true
	);
		
	// New access that combines the reduction
	#pragma oss task in(x)
	{
		std::ostringstream oss;
		oss << "Expected reduction computation when task " << fourTimesActiveCPUs + 1 <<
			" (READ) is executed";
		tap.evaluate(x == fourTimesActiveCPUs, oss.str());
	}
	
	#pragma oss taskwait
	
	tap.end();
}
