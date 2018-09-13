/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

/*
 * Test whether finishing a reduction task with nested reduction tasks handles
 * the reduction properly
 *
 */

#include <algorithm>
#include <vector>

#include <cassert>
#include <cmath>

#include <unistd.h>

#include <nanos6/debug.h>

#include <Atomic.hpp>
#include <Functors.hpp>
#include "TestAnyProtocolProducer.hpp"


#define NUM_TASKS_PER_CPU 500
#define BRANCHING_FACTOR 3
#define SUSTAIN_MICROSECONDS 10000L

#define MAX_TASKS 8000
#define MIN_TASKS_PER_CPU 16


using namespace Functors;


TestAnyProtocolProducer tap;
static Atomic<int> numTasks(0);
static int totalTasks;
static int finalDepth;
static int branchingFactor;
static double delayMultiplier = 1.0;

std::vector< Atomic<bool> > ready;

void recursion(int& x, int depth, Atomic<bool>& parentReady) {
	if (depth < finalDepth) {
		int id = ++numTasks; // Id from next reduction task
		assert (id <= totalTasks);
		
		Atomic<bool>& nextReady = ready[id];
		
		nextReady = false;
		
		#pragma oss task weakreduction(+: x) shared(parentReady, nextReady)
		{
			#pragma oss task reduction(+: x)
			{
				x++;
			}
			
			for (int i = 0; i < branchingFactor; ++i) {
				recursion(x, depth + 1, nextReady);
			}
			
			// Wait for parent task to have finished
			std::ostringstream oss;
			oss << "Task " << id << "'s parent reduction task has finished execution";
			tap.timedEvaluate(
				True< Atomic<bool> >(parentReady),
				SUSTAIN_MICROSECONDS*delayMultiplier,
				oss.str(),
				/* weak */ true
			);
			usleep(SUSTAIN_MICROSECONDS);
			
			tap.emitDiagnostic("Task ", id, "/", totalTasks,
				" (REDUCTION) is executed");
		}
		
		nextReady = true;
	}
}

int main() {
	int activeCPUs = nanos6_get_num_cpus();
	delayMultiplier = sqrt(activeCPUs);
	
	// Maximum, it gets rounded to closest complete level
	totalTasks =
		std::min(
			NUM_TASKS_PER_CPU*activeCPUs,
			std::max(MAX_TASKS, MIN_TASKS_PER_CPU*activeCPUs)
		);
	branchingFactor = BRANCHING_FACTOR;
	
	assert(totalTasks > 1);
	assert(branchingFactor > 1);
	
	// Compute depth required to instantiate at max 'totalTasks' (lower bound)
	finalDepth = log(totalTasks*(branchingFactor - 1) + 1)/log(branchingFactor);
	// Compute real aggregate total number of tasks (for any branching factor >= 2)
	totalTasks = (pow(branchingFactor, finalDepth) - 1)/(branchingFactor - 1);
	
	ready = std::vector< Atomic<bool> >(totalTasks + 1);
	
	int x = 0;
	Atomic<bool>& initReady = ready[0];
	
	initReady = true;
	
	tap.registerNewTests(totalTasks + 1);
	tap.begin();
	
	recursion(x, 0, initReady);
	
	#pragma oss taskwait
	
	std::ostringstream oss;
	oss << "Expected reduction computation when all " << numTasks << "/" << totalTasks << " tasks are executed, task nesting depth = " << finalDepth;
	tap.evaluate(x == totalTasks, oss.str());
	
	tap.end();
}
