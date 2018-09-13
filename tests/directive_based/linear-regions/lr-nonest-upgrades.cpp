/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>

#include <cassert>
#include <cstdio>
#include <sstream>

#include <string.h>
#include <unistd.h>

#include <Atomic.hpp>
#include <Functors.hpp>
#include "TestAnyProtocolProducer.hpp"
#include "Timer.hpp"


#define SUSTAIN_MICROSECONDS 100000L


using namespace Functors;


TestAnyProtocolProducer tap;


template <int NUM_TASKS>
struct ExperimentStatus {
	Atomic<bool> _taskHasStarted[NUM_TASKS];
	Atomic<bool> _taskHasFinished[NUM_TASKS];
	
	ExperimentStatus()
		: _taskHasStarted(), _taskHasFinished()
	{
		for (int i=0; i < NUM_TASKS; i++) {
			_taskHasStarted[i].store(false);
			_taskHasFinished[i].store(false);
		}
	}
};


template <int NUM_TASKS>
struct ExpectedOutcome {
	bool _taskAWaitsTaskB[NUM_TASKS][NUM_TASKS];
	
	ExpectedOutcome()
		: _taskAWaitsTaskB()
	{
		for (int i=0; i < NUM_TASKS; i++) {
			for (int j=0; j < NUM_TASKS; j++) {
				_taskAWaitsTaskB[i][j] = false;
			}
		}
	}
};


template <int NUM_TASKS>
static void dependencyVerification(int currentTaskNumber, ExperimentStatus<NUM_TASKS> &status, ExpectedOutcome<NUM_TASKS> &expected)
{
	for (int otherTaskNumber = currentTaskNumber+1; otherTaskNumber < NUM_TASKS; otherTaskNumber++) {
		std::ostringstream oss;
		
		if (expected._taskAWaitsTaskB[otherTaskNumber][currentTaskNumber]) {
			oss << "Evaluating that T" << otherTaskNumber << " does not start before T" << currentTaskNumber << " finishes";
			
			tap.sustainedEvaluate(
				False< Atomic<bool> >(status._taskHasStarted[otherTaskNumber]),
				SUSTAIN_MICROSECONDS,
				oss.str()
			);
		} else {
			oss << "Evaluating that T" << otherTaskNumber << " can finish before T" << currentTaskNumber << " finishes";
			
			tap.timedEvaluate(
				True< Atomic<bool> >(status._taskHasFinished[otherTaskNumber]),
				SUSTAIN_MICROSECONDS*2L,
				oss.str(),
				true
			);
		}
	}
}


template <int NUM_TASKS>
static inline void taskCode(int currentTaskNumber, ExperimentStatus<NUM_TASKS> &status, ExpectedOutcome<NUM_TASKS> &expected)
{
	status._taskHasStarted[currentTaskNumber] = true;
	for (int otherTaskNumber = 0; otherTaskNumber < currentTaskNumber; otherTaskNumber++) {
		std::ostringstream oss;
		
		if (expected._taskAWaitsTaskB[currentTaskNumber][otherTaskNumber]) {
			oss << "Evaluating that when T" << currentTaskNumber << " starts T" << otherTaskNumber << " has finished";
			
			tap.evaluate(
				status._taskHasFinished[otherTaskNumber].load(),
				oss.str()
			);
		} else {
			#if PERFORM_NON_GUARANTEED_CHECKS
			oss << "Evaluating that when T" << currentTaskNumber << " starts T" << otherTaskNumber << " has not finished";
			
			tap.evaluateWeak(
				!status._taskHasFinished[otherTaskNumber].load(),
				oss.str(),
				"expected independent behavior but may not always manifest"
			);
			#endif
		}
	}
	
	#if PERFORM_NON_GUARANTEED_CHECKS
	for (int otherTaskNumber = currentTaskNumber+1; otherTaskNumber < NUM_TASKS; otherTaskNumber++) {
		std::ostringstream oss;
		
		if (!expected._taskAWaitsTaskB[otherTaskNumber][currentTaskNumber]) {
			oss << "Evaluating that T" << otherTaskNumber << " can finish before T" << currentTaskNumber << " finishes";
			
			tap.timedEvaluate(
				True< Atomic<bool> >(status._taskHasFinished[otherTaskNumber]),
				SUSTAIN_MICROSECONDS,
				oss.str(),
				true
			);
		}
	}
	#endif
	
	status._taskHasFinished[currentTaskNumber] = true;
	
	for (int otherTaskNumber = currentTaskNumber+1; otherTaskNumber < NUM_TASKS; otherTaskNumber++) {
		std::ostringstream oss;
		
		if (expected._taskAWaitsTaskB[otherTaskNumber][currentTaskNumber]) {
			oss << "Evaluating that T" << otherTaskNumber << " does not start before T" << currentTaskNumber << " finishes";
			
			tap.sustainedEvaluate(
				False< Atomic<bool> >(status._taskHasStarted[otherTaskNumber]),
				SUSTAIN_MICROSECONDS,
				oss.str()
			);
		}
	}
}


static inline int numSubtests(int numTasks, int numWaits)
{
	#if PERFORM_NON_GUARANTEED_CHECKS
		return numTasks * (numTasks - 1);
	#else
		return numWaits * 2;
	#endif
}


int main(int argc, char **argv)
{
	nanos6_wait_for_full_initialization();
	
	long activeCPUs = nanos6_get_num_cpus();
	if (activeCPUs < 4) {
		// This test only works correctly with at least 4 CPUs
		tap.registerNewTests(1);
		tap.begin();
		tap.skip("This test does not work with less than 4 CPUs");
		tap.end();
		return 0;
	}
	
	tap.registerNewTests(
		numSubtests(4, 1)
		+ numSubtests(4, 2)
		+ numSubtests(3, 1)
		+ numSubtests(3, 1)
		+ numSubtests(4, 2)
		+ numSubtests(5, 2)
	);
	
	tap.begin();
	
	int var[8];
	
	
	// Test 1
	{
		tap.emitDiagnostic("Test 1:   I0 I0 I0 I0 I0 I0 I0 I0");
		tap.emitDiagnostic("Test 1:         O0 O0 O0 O0      ");
		tap.emitDiagnostic("Test 1:   I1 I1 I2 I2 I2 I2 I3 I3");
		
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[2][0] = true;
		
		#pragma oss task in(var[0;8]) out(var[2;4]) label(I0 I0 I0 I0 I0 I0 I0 I0 + __ __ O0 O0 O0 O0 __ __)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task  in(var[0;2]) label(I1 I1 __ __ __ __ __ __)  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task  in(var[2;4]) label(__ __ I2 I2 I2 I2 __ __)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task  in(var[6;2]) label(__ __ __ __ __ __ I3 I3)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 2
	{
		tap.emitDiagnostic("Test 2:         I0 I0 I0 I0 I0 I0");
		tap.emitDiagnostic("Test 2:   O0 O0 O0 O0            ");
		tap.emitDiagnostic("Test 2:   I1 I1 I2 I2 I3 I3 I3 I3");
		
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[1][0] = true;
		expected._taskAWaitsTaskB[2][0] = true;
		
		#pragma oss task in(var[2;6]) out(var[0;4]) label(__ __ I0 I0 I0 I0 I0 I0 + O0 O0 O0 O0 __ __ __ __)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task  in(var[0;2]) label(I1 I1 __ __ __ __ __ __)  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task  in(var[2;2]) label(__ __ I2 I2 __ __ __ __)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task  in(var[4;4]) label(__ __ __ __ I3 I3 I3 I3)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 3
	{
		tap.emitDiagnostic("Test 3:   I0 I0 I0 I0 I0 I0 I0 I0");
		tap.emitDiagnostic("Test 3:                     O0 O0");
		tap.emitDiagnostic("Test 3:   I1 I1 I1 I1 I2 I2 I2 I2");
		
		ExperimentStatus<3> status;
		ExpectedOutcome<3> expected;
		expected._taskAWaitsTaskB[2][0] = true;
		
		#pragma oss task in(var[0;8]) out(var[6;2]) label(I0 I0 I0 I0 I0 I0 I0 I0 + __ __ __ __ __ __ O0 O0)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task  in(var[0;4]) label(I1 I1 I1 I1 __ __ __ __)  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task  in(var[4;4]) label(__ __ __ __ I2 I2 I2 I2)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 4
	{
		tap.emitDiagnostic("Test 4:   I0 I0 I0 I0 I0 I0      ");
		tap.emitDiagnostic("Test 4:               O0 O0      ");
		tap.emitDiagnostic("Test 4:   I2 I2 I1 I1 I1 I1 I1 I1");
		
		ExperimentStatus<3> status;
		ExpectedOutcome<3> expected;
		expected._taskAWaitsTaskB[1][0] = true;
		
		#pragma oss task in(var[0;6]) out(var[4;2]) label(I0 I0 I0 I0 I0 I0 __ __ + __ __ __ __ O0 O0 __ __)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task  in(var[2;6]) label(__ __ I1 I1 I1 I1 I1 I1)  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task  in(var[0;2]) label(I2 I2 __ __ __ __ __ __)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 5
	{
		tap.emitDiagnostic("Test 5:   I0 I0 I0 I0 I0 I0      ");
		tap.emitDiagnostic("Test 5:               I1 I1 I1 I1");
		tap.emitDiagnostic("Test 5:               O1 O1 O1 O1");
		tap.emitDiagnostic("Test 5:   I2 I2 I2 I2            ");
		tap.emitDiagnostic("Test 5:         I3 I3 I3 I3      ");
		
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[1][0] = true;
		expected._taskAWaitsTaskB[3][1] = true;
		
		#pragma oss task  in(var[0;6]) label(I0 I0 I0 I0 I0 I0 __ __)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task in(var[4;4]) out(var[4;4]) label(__ __ __ __ I1 I1 I1 I1+ __ __ __ __ O1 O1 O1 O1)  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task  in(var[0;4]) label(I2 I2 I2 I2 __ __ __ __)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task  in(var[2;4]) label(__ __ I3 I3 I3 I3 __ __)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 6
	{
		tap.emitDiagnostic("Test 6:   I0 I0 I0 I0 I0 I0 I0 I0");
		tap.emitDiagnostic("Test 6:            O0 O0         ");
		tap.emitDiagnostic("Test 6:         I1 I1 I1 I1      ");
		tap.emitDiagnostic("Test 6:   I3 I2 I2 I2 I2 I2 I2 I4");
		
		ExperimentStatus<5> status;
		ExpectedOutcome<5> expected;
		expected._taskAWaitsTaskB[1][0] = true;
		expected._taskAWaitsTaskB[2][0] = true;
		
		#pragma oss task in(var[0;8]) out(var[3;2]) label(I0 I0 I0 I0 I0 I0 I0 I0 + __ __ __ O0 O0 __ __ __)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task  in(var[2;4]) label(__ __ I1 I1 I1 I1 __ __)  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task  in(var[1;6]) label(__ I2 I2 I2 I2 I2 I2 __)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task  in(var[0;1]) label(I3 __ __ __ __ __ __ __)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss task  in(var[7;1]) label(__ __ __ __ __ __ __ I4)  shared(status, expected)
		taskCode(4, status, expected);
		
		#pragma oss taskwait
	}
	
	tap.end();
	
	return 0;
}

