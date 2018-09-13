/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

/*
 * Test whether a single nested reduction handles intermediate taskwait clauses properly
 *
 */

#include <atomic>

#include <assert.h>
#include <math.h>
#include <unistd.h>
#include <stdio.h>

#include <nanos6/debug.h>

#include <Atomic.hpp>
#include "TestAnyProtocolProducer.hpp"


#define NUM_TASKS_PER_CPU 500
#define BRANCHING_FACTOR 3
#define SUSTAIN_MICROSECONDS 1000000L


TestAnyProtocolProducer tap;
static Atomic<int> numTasks(0);
static int totalTasks;
static int finalDepth;
static int branchingFactor;
static double delayMultiplier = 1.0;

void recursion(int& x, int depth) {
	if (depth < finalDepth) {
		#pragma oss task reduction(+: x)
		{
			int id = ++numTasks; // Id from next reduction task
			x++;
			
			for (int i = 0; i < branchingFactor; ++i) {
				recursion(x, depth + 1);
			}
			
			#pragma oss taskwait
			
			int expected = 1;
			for (int i = finalDepth - depth; i > 1; i--) expected *= BRANCHING_FACTOR;
			tap.emitDiagnostic("Task ", id, "/", totalTasks,
				" local x=", x, " expected=", expected, " good=", (x == expected));
		}
	}
}

int main() {
	long activeCPUs = nanos6_get_num_cpus();
	delayMultiplier = sqrt(activeCPUs);
	
#if TEST_LESS_THREADS
	totalTasks = std::min(NUM_TASKS_PER_CPU*activeCPUs, 1000); // Maximum, it gets rounded to closest complete level
#else
	totalTasks = NUM_TASKS_PER_CPU*activeCPUs; // Maximum, it gets rounded to closest complete level
#endif
	branchingFactor = BRANCHING_FACTOR;
	
	assert(totalTasks > 1);
	assert(branchingFactor > 1);
	
	// Compute depth required to instantiate at max 'totalTasks' (lower bound)
	finalDepth = log(totalTasks*(branchingFactor - 1) + 1)/log(branchingFactor);
	// Compute real aggregate total number of tasks (for any branching factor >= 2)
	totalTasks = (pow(branchingFactor, finalDepth) - 1)/(branchingFactor - 1);
	
	int x = 0;
	
	tap.registerNewTests(1);
	tap.begin();
	
	recursion(x, 0);
	
	#pragma oss taskwait
	std::ostringstream oss;
	oss << "After a taskwait, the reduction variable contains the effect of " << x << " out of " << totalTasks << " tasks with nesting depth " << finalDepth;
	tap.evaluate(x == totalTasks, oss.str());
	
	tap.end();
}
