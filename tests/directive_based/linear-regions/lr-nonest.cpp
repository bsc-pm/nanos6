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
		numSubtests(4, 5)
		+ numSubtests(7, 16)
		+ numSubtests(7, 14)
		+ numSubtests(8, 10)
		+ numSubtests(4, 6)
		+ numSubtests(4, 5)
		+ numSubtests(5, 4)
		+ numSubtests(5, 4)
	);
	
	tap.begin();
	
	int var[8];
	
	
	// Test 1
	{
		tap.emitDiagnostic("Test 1:   O0 O0 O0 O0 O0 O0 O0 O0");
		tap.emitDiagnostic("Test 1:   I1 I1 I1 I1 I2 I2 I2 I2");
		tap.emitDiagnostic("Test 1:   O3 O3 O3 O3 O3 O3 O3 O3");
		
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[1][0] = true;
		expected._taskAWaitsTaskB[2][0] = true;
		expected._taskAWaitsTaskB[3][0] = true; expected._taskAWaitsTaskB[3][1] = true; expected._taskAWaitsTaskB[3][2] = true;
		
		#pragma oss task out(var[0;8]) label(O0 O0 O0 O0 O0 O0 O0 O0)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task  in(var[0;4]) label(I1 I1 I1 I1 __ __ __ __)  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task  in(var[4;4]) label(__ __ __ __ I2 I2 I2 I2)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task out(var[0;8]) label(O3 O3 O3 O3 O3 O3 O3 O3)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 2
	{
		tap.emitDiagnostic("Test 2:   O0 O0 O0 O0 O1 O1 O1 O1");
		tap.emitDiagnostic("Test 2:   I2 I2 I2 I2 I2 I2 I2 I2");
		tap.emitDiagnostic("Test 2:   O3 O3 O3 O3 O4 O4 O4 O4");
		tap.emitDiagnostic("Test 2:   I5 I5 I5 I5 I6 I6 I6 I6");
		
		ExperimentStatus<7> status;
		ExpectedOutcome<7> expected;
		expected._taskAWaitsTaskB[2][0] = true; expected._taskAWaitsTaskB[2][1] = true;
		expected._taskAWaitsTaskB[3][0] = true; expected._taskAWaitsTaskB[3][1] = true; expected._taskAWaitsTaskB[3][2] = true;
		expected._taskAWaitsTaskB[4][0] = true; expected._taskAWaitsTaskB[4][1] = true; expected._taskAWaitsTaskB[4][2] = true;
		expected._taskAWaitsTaskB[5][0] = true; expected._taskAWaitsTaskB[5][1] = true; expected._taskAWaitsTaskB[5][2] = true; expected._taskAWaitsTaskB[5][3] = true;
		expected._taskAWaitsTaskB[6][0] = true; expected._taskAWaitsTaskB[6][1] = true; expected._taskAWaitsTaskB[6][2] = true; expected._taskAWaitsTaskB[6][4] = true;
		
		#pragma oss task out(var[0;4]) label(O0 O0 O0 O0 __ __ __ __)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task out(var[4;4]) label(__ __ __ __ O1 O1 O1 O1)  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task  in(var[0;8]) label(I2 I2 I2 I2 I2 I2 I2 I2)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task out(var[0;4]) label(O3 O3 O3 O3 __ __ __ __)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss task out(var[4;4]) label(__ __ __ __ O4 O4 O4 O4)  shared(status, expected)
		taskCode(4, status, expected);
		
		#pragma oss task  in(var[0;4]) label(I5 I5 I5 I5 __ __ __ __)  shared(status, expected)
		taskCode(5, status, expected);
		
		#pragma oss task  in(var[4;4]) label(__ __ __ __ I6 I6 I6 I6)  shared(status, expected)
		taskCode(6, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 3
	{
		tap.emitDiagnostic("Test 3:   O0 O0 O0 O0 O1 O1 O1 O1");
		tap.emitDiagnostic("Test 3:   I2 I2 I3 I3 I3 I3 I3 I3");
		tap.emitDiagnostic("Test 3:   O4 O4 O4 O4 I5 I5 I5 I5");
		tap.emitDiagnostic("Test 3:   O6 O6 O6 O6 O6 O6 O6 O6");
		
		ExperimentStatus<7> status;
		ExpectedOutcome<7> expected;
		expected._taskAWaitsTaskB[2][0] = true;
		expected._taskAWaitsTaskB[3][0] = true; expected._taskAWaitsTaskB[3][1] = true;
		expected._taskAWaitsTaskB[4][0] = true; expected._taskAWaitsTaskB[4][1] = true; expected._taskAWaitsTaskB[4][2] = true; expected._taskAWaitsTaskB[4][3] = true;
		expected._taskAWaitsTaskB[5][1] = true;
		expected._taskAWaitsTaskB[6][0] = true; expected._taskAWaitsTaskB[6][1] = true; expected._taskAWaitsTaskB[6][2] = true; expected._taskAWaitsTaskB[6][3] = true; expected._taskAWaitsTaskB[6][4] = true; expected._taskAWaitsTaskB[6][5] = true;
		
		#pragma oss task out(var[0;4]) label(O0 O0 O0 O0 __ __ __ __)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task out(var[4;4]) label(__ __ __ __ O1 O1 O1 O1)  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task  in(var[0;2]) label(I2 I2 __ __ __ __ __ __)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task  in(var[2;6]) label(__ __ I3 I3 I3 I3 I3 I3)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss task out(var[0;4]) label(O4 O4 O4 O4 __ __ __ __)  shared(status, expected)
		taskCode(4, status, expected);
		
		#pragma oss task  in(var[4;4]) label(__ __ __ __ I5 I5 I5 I5)  shared(status, expected)
		taskCode(5, status, expected);
		
		#pragma oss task out(var[0;8]) label(O6 O6 O6 O6 O6 O6 O6 O6)  shared(status, expected)
		taskCode(6, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 4
	{
		tap.emitDiagnostic("Test 4:   O0 O0 O0 O0 O0 O0 O0 O0");
		tap.emitDiagnostic("Test 4:   I1 I1 I2 I2 I2 I2 I3 I3");
		tap.emitDiagnostic("Test 4:         O4 O4 O4 O4      ");
		tap.emitDiagnostic("Test 4:   I5 I5 I6 I6 I6 I6 I7 I7");
		
		ExperimentStatus<8> status;
		ExpectedOutcome<8> expected;
		expected._taskAWaitsTaskB[1][0] = true;
		expected._taskAWaitsTaskB[2][0] = true;
		expected._taskAWaitsTaskB[3][0] = true;
		expected._taskAWaitsTaskB[4][0] = true; expected._taskAWaitsTaskB[4][2] = true;
		expected._taskAWaitsTaskB[5][0] = true;
		expected._taskAWaitsTaskB[6][0] = true; expected._taskAWaitsTaskB[6][2] = true; expected._taskAWaitsTaskB[6][4] = true;
		expected._taskAWaitsTaskB[7][0] = true;
		
		#pragma oss task out(var[0;8]) label(O0 O0 O0 O0 O0 O0 O0 O0)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task  in(var[0;2]) label(I1 I1__ __ __ __ __ __ )  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task  in(var[2;4]) label(__ __ I2 I2 I2 I2 __ __)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task  in(var[6;2]) label(__ __ __ __ __ __ I3 I3)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss task out(var[2;4]) label(__ __ O4 O4 O4 O4 __ __)  shared(status, expected)
		taskCode(4, status, expected);
		
		#pragma oss task  in(var[0;2]) label(I5 I5 __ __ __ __ __ __)  shared(status, expected)
		taskCode(5, status, expected);
		
		#pragma oss task  in(var[2;4]) label(__ __ I6 I6 I6 I6 __ __)  shared(status, expected)
		taskCode(6, status, expected);
		
		#pragma oss task  in(var[6;2]) label(__ __ __ __ __ __ I7 I7)  shared(status, expected)
		taskCode(7, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 5
	{
		tap.emitDiagnostic("Test 5:   O0 O0 O0 O0 O0 O0 O0 O0");
		tap.emitDiagnostic("Test 5:            I1 I1         ");
		tap.emitDiagnostic("Test 5:         O2 O2 O2 O2      ");
		tap.emitDiagnostic("Test 5:   I3 I3 I3 I3 I3 I3 I3 I3");
		
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[1][0] = true;
		expected._taskAWaitsTaskB[2][0] = true; expected._taskAWaitsTaskB[2][1] = true;
		expected._taskAWaitsTaskB[3][0] = true; expected._taskAWaitsTaskB[3][1] = true; expected._taskAWaitsTaskB[3][2] = true;
		
		#pragma oss task out(var[0;8]) label(O0 O0 O0 O0 O0 O0 O0 O0)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task  in(var[3;2]) label(__ __ __ I1 I1 __ __ __)  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task out(var[2;4]) label(__ __ O2 O2 O2 O2 __ __)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task  in(var[0;8]) label(I3 I3 I3 I3 I3 I3 I3 I3)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 6
	{
		tap.emitDiagnostic("Test 6:   O1 O1 O1 O1 O0 O0 O0 O0");
		tap.emitDiagnostic("Test 6:         I2 I2 I2 I2      ");
		tap.emitDiagnostic("Test 6:   O3 O3 O3 O3 O3 O3 O3 O3");
		
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[2][0] = true; expected._taskAWaitsTaskB[2][1] = true;
		expected._taskAWaitsTaskB[3][0] = true; expected._taskAWaitsTaskB[3][1] = true; expected._taskAWaitsTaskB[3][2] = true;
		
		#pragma oss task out(var[4;4]) label(__ __ __ __ O0 O0 O0 O0)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task out(var[0;4]) label(O1 O1 O1 O1 __ __ __ __)  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task  in(var[2;4]) label(__ __ I2 I2 I2 I2 __ __)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task out(var[0;8]) label(O3 O3 O3 O3 O3 O3 O3 O3)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 7
	{
		tap.emitDiagnostic("Test 7:   O1 O1 O1 O1 O0 O0 O0 O0");
		tap.emitDiagnostic("Test 7:   I2 I2 I2 I2 I2 I2 I2 I2");
		tap.emitDiagnostic("Test 7:   I4 I4 I3 I3            ");
		
		ExperimentStatus<5> status;
		ExpectedOutcome<5> expected;
		expected._taskAWaitsTaskB[2][0] = true; expected._taskAWaitsTaskB[2][1] = true;
		expected._taskAWaitsTaskB[3][1] = true;
		expected._taskAWaitsTaskB[4][1] = true;
		
		#pragma oss task out(var[4;4]) label(__ __ __ __ O0 O0 O0 O0)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task out(var[0;4]) label(O1 O1 O1 O1 __ __ __ __)  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task  in(var[0;8]) label(I2 I2 I2 I2 I2 I2 I2 I2)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task  in(var[2;2]) label(__ __ I3 I3 __ __ __ __)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss task  in(var[0;2]) label(I4 I4 __ __ __ __ __ __)  shared(status, expected)
		taskCode(4, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 8
	{
		tap.emitDiagnostic("Test 8:   I1 I1 I1 I1 I0 I0 I0 I0");
		tap.emitDiagnostic("Test 8:   I2 I2 I2 I2 I2 I2 I2 I2");
		tap.emitDiagnostic("Test 8:   O3 O3 O3 O3 O4 O4 O4 O4");
		
		ExperimentStatus<5> status;
		ExpectedOutcome<5> expected;
		expected._taskAWaitsTaskB[3][1] = true; expected._taskAWaitsTaskB[3][2] = true;
		expected._taskAWaitsTaskB[4][0] = true; expected._taskAWaitsTaskB[4][2] = true;
		
		#pragma oss task  in(var[4;4]) label(__ __ __ __ I0 I0 I0 I0)  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task  in(var[0;4]) label(I1 I1 I1 I1 __ __ __ __)  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task  in(var[0;8]) label(I2 I2 I2 I2 I2 I2 I2 I2)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task out(var[0;4]) label(O3 O3 O3 O3 __ __ __ __)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss task out(var[4;4]) label(__ __ __ __ O4 O4 O4 O4)  shared(status, expected)
		taskCode(4, status, expected);
		
		#pragma oss taskwait
	}
	
	tap.end();
	
	return 0;
}

