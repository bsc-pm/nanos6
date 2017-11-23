/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

/*
 * Test that stresses the system by registering many simultaneous reductions of
 * multiple tasks each
 *
 */

#include <Atomic.hpp>
#include "TestAnyProtocolProducer.hpp"


#define N_REDUCTIONS 500
#define TASKS_PER_REDUCTION 500

#define DIAGNOSTIC_PROGRESS_LINES 10 // Number of output diagnostic lines for reduction submitting


TestAnyProtocolProducer tap;
Atomic<int> numReductions(0);

int main() {
	int array[N_REDUCTIONS] = { 0 };
	
	int outputStep = N_REDUCTIONS/DIAGNOSTIC_PROGRESS_LINES;
	
	tap.registerNewTests(1);
	tap.begin();
	
	for (unsigned int i = 0; i < N_REDUCTIONS; ++i) {
		int id = ++numReductions;
		
		int& element = array[i];
		for (unsigned int task = 0; task < TASKS_PER_REDUCTION; ++task) {
			#pragma oss task reduction(+: element)
			{
				element++;
			}
		}
		
		if (id > 0 && id%outputStep == 0)
			tap.emitDiagnostic("Reductions ", id - (outputStep - 1), "-", id, "/", N_REDUCTIONS,
				" submitted");
	}

	#pragma oss taskwait
	
	bool correct = true;
	for (size_t i = 0; i < N_REDUCTIONS; ++i) {
		if (array[i] != TASKS_PER_REDUCTION) {
			correct = false;
			break;
		}
	}
	
	tap.evaluate(correct, "All reductions have the expected result");
	
	tap.end();
}
