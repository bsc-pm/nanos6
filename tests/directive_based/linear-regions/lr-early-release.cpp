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
	Atomic<bool> _taskHasReallyFinished[NUM_TASKS];
	
	ExperimentStatus()
		: _taskHasStarted(), _taskHasFinished()
	{
		for (int i=0; i < NUM_TASKS; i++) {
			_taskHasStarted[i].store(false);
			_taskHasFinished[i].store(false);
			_taskHasReallyFinished[i].store(false);
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
static inline void taskPrologue(int currentTaskNumber, ExperimentStatus<NUM_TASKS> &status, ExpectedOutcome<NUM_TASKS> &expected, int waitForTaskToFinish = -1)
{
	status._taskHasStarted[currentTaskNumber] = true;
	
	if (waitForTaskToFinish != -1) {
		while (!status._taskHasReallyFinished[waitForTaskToFinish].load()) {
		}
	}
	
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
}


template <int NUM_TASKS>
static inline void taskEpilogue(int currentTaskNumber, ExperimentStatus<NUM_TASKS> &status, ExpectedOutcome<NUM_TASKS> &expected)
{
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
	
	status._taskHasReallyFinished[currentTaskNumber] = true;
}


template <int NUM_TASKS>
static inline void taskCode(int currentTaskNumber, ExperimentStatus<NUM_TASKS> &status, ExpectedOutcome<NUM_TASKS> &expected, int waitForTaskToFinish = -1)
{
	taskPrologue(currentTaskNumber, status, expected, waitForTaskToFinish);
	taskEpilogue(currentTaskNumber, status, expected);
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
	if (activeCPUs < 2) {
		// This test only works correctly with at least 2 CPUs
		tap.registerNewTests(1);
		tap.begin();
		tap.skip("This test does not work with less than 2 CPUs");
		tap.end();
		return 0;
	}
	
	tap.registerNewTests(
		numSubtests(5, 2)
		+ numSubtests(4, 2)
		+ numSubtests(5, 5)
		+ numSubtests(3, 2)
		+ numSubtests(7, 8)
		+ numSubtests(9, 14)
	);
	
	tap.begin();
	
	int var[8];
	
	
	// Test 1
	{
		tap.emitDiagnostic("Test 1:   O0 O0 O0 O0 O0 O0 O0 O0");
		tap.emitDiagnostic("Test 1:         O1 O1 O1 O1       (son of 0)");
		tap.emitDiagnostic("Test 1:   I2 I2 I3 I3 I3 I3 I4 I4");
		
		ExperimentStatus<5> status;
		ExpectedOutcome<5> expected;
		expected._taskAWaitsTaskB[3][0] = true; expected._taskAWaitsTaskB[3][1] = true;
		
		#pragma oss task out(var[0;8])    label(O0 O0 O0 O0 O0 O0 O0 O0)  shared(status, expected)
		{
			taskPrologue(0, status, expected);
			
			#pragma oss task out(var[2;4]) label(__ __ O1 O1 O1 O1 __ __)  shared(status, expected)
			taskCode(1, status, expected, 0);
			
			taskEpilogue(0, status, expected);
		}
		
		#pragma oss task  in(var[0;2])    label(I2 I2 __ __ __ __ __ __)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task  in(var[2;4])    label(__ __ I3 I3 I3 I3 __ __)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss task  in(var[6;2])    label(__ __ __ __ __ __ I4 I4)  shared(status, expected)
		taskCode(4, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 2
	{
		tap.emitDiagnostic("Test 2:   O0 O0 O0 O0 O0 O0 O0 O0");
		tap.emitDiagnostic("Test 2:         O1 O1 O1 O1       (son of 0)");
		tap.emitDiagnostic("Test 2:   I2 I2 I2 I2 I3 I3 I3 I3");
		
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[2][1] = true;
		expected._taskAWaitsTaskB[3][1] = true;
		
		#pragma oss task out(var[0;8])    label(O0 O0 O0 O0 O0 O0 O0 O0)  shared(status, expected)
		{
			taskPrologue(0, status, expected);
			
			#pragma oss task out(var[2;4]) label(__ __ O1 O1 O1 O1 __ __)  shared(status, expected)
			taskCode(1, status, expected, 0);
			
			taskEpilogue(0, status, expected);
		}
		
		#pragma oss task  in(var[0;4])    label(I2 I2 I2 I2 __ __ __ __)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task  in(var[4;4])    label(__ __ __ __ I3 I3 I3 I3)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 3
	{
		tap.emitDiagnostic("Test 3:   O0 O0 O0 O0 O0 O0 O0 O0");
		tap.emitDiagnostic("Test 3:   O1 O1             O2 O2 (sons of 0)");
		tap.emitDiagnostic("Test 3:   I3 I3 I3 I4 I4 I5 I5 I5");
		
		ExperimentStatus<6> status;
		ExpectedOutcome<6> expected;
		expected._taskAWaitsTaskB[3][0] = true; expected._taskAWaitsTaskB[3][1] = true;
		expected._taskAWaitsTaskB[4][0] = true;
		expected._taskAWaitsTaskB[5][0] = true; expected._taskAWaitsTaskB[5][2] = true;
		
		#pragma oss task out(var[0;8])    label(O0 O0 O0 O0 O0 O0 O0 O0)  shared(status, expected)
		{
			taskPrologue(0, status, expected);
			
			#pragma oss task out(var[0;2]) label(O1 O1 __ __ __ __ __ __)  shared(status, expected)
			taskCode(1, status, expected, 0);
			
			#pragma oss task out(var[6;2]) label(__ __ __ __ __ __ O2 O2)  shared(status, expected)
			taskCode(2, status, expected, 0);
			
			taskEpilogue(0, status, expected);
		}
		
		#pragma oss task  in(var[0;3])    label(I3 I3 I3 __ __ __ __ __)  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss task  in(var[3;2])    label(__ __ __ I4 I4 __ __ __)  shared(status, expected)
		taskCode(4, status, expected);
		
		#pragma oss task  in(var[5;3])    label(__ __ __ __ __ I5 I5 I5)  shared(status, expected)
		taskCode(5, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 4
	{
		tap.emitDiagnostic("Test 4:   O0 O0 O0 O0 O0 O0 O0 O0");
		tap.emitDiagnostic("Test 4:         O1 O1 O1 O1       (son of 0)");
		tap.emitDiagnostic("Test 4:      I2 I2 I2 I2 I2 I2   ");
		
		ExperimentStatus<3> status;
		ExpectedOutcome<3> expected;
		expected._taskAWaitsTaskB[2][0] = true; expected._taskAWaitsTaskB[2][1] = true;
		
		#pragma oss task out(var[0;8])    label(O0 O0 O0 O0 O0 O0 O0 O0)  shared(status, expected)
		{
			taskPrologue(0, status, expected);
			
			#pragma oss task out(var[2;4]) label(__ __ O1 O1 O1 O1 __ __)  shared(status, expected)
			taskCode(1, status, expected, 0);
			
			taskEpilogue(0, status, expected);
		}
		
		#pragma oss task  in(var[1;6])    label(__ I2 I2 I2 I2 I2 I2 __)  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 5
	{
		tap.emitDiagnostic("Test 5:   I0 I0 I0 I0 I0 I0 I0 I0");
		tap.emitDiagnostic("Test 5:   I1 I1 I1 I1 I1 I1 I1 I1");
		tap.emitDiagnostic("Test 5:   I2 I2             I3 I3 (sons of 1)");
		tap.emitDiagnostic("Test 5:   O6 O6 O5 O5 O5 O5 O4 O4");
		
		ExperimentStatus<7> status;
		ExpectedOutcome<7> expected;
		expected._taskAWaitsTaskB[4][0] = true; expected._taskAWaitsTaskB[4][1] = true; expected._taskAWaitsTaskB[4][3] = true;
		expected._taskAWaitsTaskB[5][0] = true; expected._taskAWaitsTaskB[5][1] = true;
		expected._taskAWaitsTaskB[6][0] = true; expected._taskAWaitsTaskB[6][1] = true; expected._taskAWaitsTaskB[6][2] = true;
		
		#pragma oss task  in(var[0;8])   label(I0 I0 I0 I0 I0 I0 I0 I0)  shared(status, expected)
		taskCode(0, status, expected, 1);
		
		#pragma oss task  in(var[0;8])   label(I1 I1 I1 I1 I1 I1 I1 I1)  shared(status, expected)
		{
			taskPrologue(1, status, expected);
			
			#pragma oss task in(var[0;2]) label(I2 I2 __ __ __ __ __ __)  shared(status, expected)
			taskCode(2, status, expected, 1);
			
			#pragma oss task in(var[6;2]) label(__ __ __ __ __ __ I3 I3)  shared(status, expected)
			taskCode(3, status, expected, 1);
			
			taskEpilogue(1, status, expected);
		}
		
		#pragma oss task out(var[6;2])   label(__ __ __ __ __ __ O4 O4)  shared(status, expected)
		taskCode(4, status, expected);
		
		#pragma oss task out(var[2;4])   label(__ __ O5 O5 O5 O5 __ __)  shared(status, expected)
		taskCode(5, status, expected);
		
		#pragma oss task out(var[0;2])   label(O6 O6 __ __ __ __ __ __)  shared(status, expected)
		taskCode(6, status, expected);
		
		#pragma oss taskwait
	}
	
	
	// Test 6
	{
		tap.emitDiagnostic("Test 6:   O0 O0 O0 O0 O0 O0 O0 O0");
		tap.emitDiagnostic("Test 6:   O1 O1 O1 O1 O2 O2 O2 O2 (sons of 0)");
		tap.emitDiagnostic("Test 6:            O3             (sons of 1)");
		tap.emitDiagnostic("Test 6:               O4          (sons of 2)");
		tap.emitDiagnostic("Test 6:   I5 I5 I5 I6 I7 I8 I8 I8");
		tap.emitDiagnostic("Test 6:   I9 I9 I9 I9 I9 I9 I9 I9");
		
		ExperimentStatus<9> status;
		ExpectedOutcome<9> expected;
		expected._taskAWaitsTaskB[5][0] = true; expected._taskAWaitsTaskB[5][1] = true;
		expected._taskAWaitsTaskB[6][0] = true; expected._taskAWaitsTaskB[6][1] = true; expected._taskAWaitsTaskB[6][2] = true; expected._taskAWaitsTaskB[6][3] = true; expected._taskAWaitsTaskB[6][4] = true;
		expected._taskAWaitsTaskB[7][0] = true; expected._taskAWaitsTaskB[7][2] = true;
		expected._taskAWaitsTaskB[8][0] = true; expected._taskAWaitsTaskB[8][1] = true; expected._taskAWaitsTaskB[8][2] = true; expected._taskAWaitsTaskB[8][3] = true; expected._taskAWaitsTaskB[8][4] = true;
		
		#pragma oss task out(var[0;8])       label(O0 O0 O0 O0 O0 O0 O0 O0)  shared(status, expected)
		{
			taskPrologue(0, status, expected);
			
			#pragma oss task out(var[0;4])    label(O1 O1 O1 O1 __ __ __ __)  shared(status, expected)
			{
				taskPrologue(1, status, expected, 0);
				
				#pragma oss task out(var[3;1]) label(__ __ __ O3 __ __ __ __)  shared(status, expected)
				taskCode(3, status, expected, 1);
				
				taskEpilogue(1, status, expected);
			}
			
			
			#pragma oss task out(var[4;4])    label(__ __ __ __ O2 O2 O2 O2)  shared(status, expected)
			{
				taskPrologue(2, status, expected, 0);
				
				#pragma oss task out(var[4;1]) label(__ __ __ __ O4 __ __ __)  shared(status, expected)
				taskCode(4, status, expected, 2);
				
				taskEpilogue(2, status, expected);
			}
			
			taskEpilogue(0, status, expected);
		}
		
		#pragma oss task  in(var[0;3])       label(I5 I5 I5 __ __ __ __ __)  shared(status, expected)
		taskCode(5, status, expected);
		
		#pragma oss task  in(var[3;2])       label(__ __ __ I6 I6 __ __ __)  shared(status, expected)
		taskCode(6, status, expected);
		
		#pragma oss task  in(var[5;3])       label(__ __ __ __ __ I7 I7 I7)  shared(status, expected)
		taskCode(7, status, expected);
		
		#pragma oss task  in(var[0;8])       label(I8 I8 I8 I8 I8 I8 I8 I8)  shared(status, expected)
		taskCode(8, status, expected);
		
		#pragma oss taskwait
	}
	
	tap.end();
	
	return 0;
}

