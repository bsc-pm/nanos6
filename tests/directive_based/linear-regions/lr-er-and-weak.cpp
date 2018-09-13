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
	if (activeCPUs < 4) {
		// This test only works correctly with at least 4 CPUs
		tap.registerNewTests(1);
		tap.begin();
		tap.skip("This test does not work with less than 4 CPUs");
		tap.end();
		return 0;
	}
	
	tap.registerNewTests(
		numSubtests(4, 2)
		+ numSubtests(4, 2)
		+ numSubtests(8, 3)
	);
	
	tap.begin();
	
	int var[8];
	
	
	// Test 1
	{
		tap.emitDiagnostic("Test 1:   i0 i0 i0 i0 i0 i0      ");
		tap.emitDiagnostic("Test 1:      I1 I1 I1 I1          (son of 0)");
		tap.emitDiagnostic("Test 1:         o2 o2 o2 o2 o2 o2");
		tap.emitDiagnostic("Test 1:            O3             (son of 2)");
		
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[3][0] = true; expected._taskAWaitsTaskB[3][1] = true;
		
		#pragma oss task  weakin(var[0;6])    label(i0 i0 i0 i0 i0 i0 __ __)  shared(status, expected)
		{
			taskPrologue(0, status, expected);
			
			#pragma oss task  in(var[1;4])     label(__ I1 I1 I1 I1 __ __ __)  shared(status, expected)
			taskCode(1, status, expected, 0);
			
			taskEpilogue(0, status, expected);
		}
		
		#pragma oss task weakout(var[2;6])    label(__ __ o2 o2 o2 o2 o2 o2)  shared(status, expected)
		{
			taskPrologue(2, status, expected);
			
			#pragma oss task out(var[3;1])     label(__ __ __ W3 __ __ __ __)  shared(status, expected)
			taskCode(3, status, expected, 2);
			
			taskEpilogue(2, status, expected);
		}
		
		#pragma oss taskwait
	}
	
	
	// Test 2
	{
		tap.emitDiagnostic("Test 2:   o0 o0 o0 o0 o0 o0 o0 o0");
		tap.emitDiagnostic("Test 2:         O1 O1 O1 O1 O1    (son of 0)");
		tap.emitDiagnostic("Test 2:      i2 i2 i2 i2 i2      ");
		tap.emitDiagnostic("Test 2:      I3 I3                (son of 2)");
		
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[3][0] = true; expected._taskAWaitsTaskB[3][1] = true;
		
		#pragma oss task weakout(var[0;8])    label(o0 o0 o0 o0 o0 o0 o0 o0)  shared(status, expected)
		{
			taskPrologue(0, status, expected);
			
			#pragma oss task out(var[2;5])     label(__ __ O1 O1 O1 O1 O1 __)  shared(status, expected)
			taskCode(1, status, expected, 0);
			
			taskEpilogue(0, status, expected);
		}
		
		#pragma oss task  weakin(var[1;5])    label(__ i2 i2 i2 i2 i2 __ __)  shared(status, expected)
		{
			taskPrologue(2, status, expected);
			
			#pragma oss task  in(var[1;2])     label(__ I3 I3 __ __ __ __ __)  shared(status, expected)
			taskCode(3, status, expected, 2);
			
			taskEpilogue(2, status, expected);
		}
		
		#pragma oss taskwait
	}
	
	
	// Test 3
	{
		tap.emitDiagnostic("Test 3:    i0  i0  i0  i0  i0  i0  i0  i0");
		tap.emitDiagnostic("Test 3:            I1  I1  I1  I1         (son of 0)");
		tap.emitDiagnostic("Test 3:   io2 io2 io2 io2 io2 io2 io2 io2");
		tap.emitDiagnostic("Test 3:    I3  I3  I3  I3  I3  I3  I3  I3 (son of 2)");
		tap.emitDiagnostic("Test 3:    I4  I4  I4  I4  I4  I4  O5  O5 (sons of 2)");
		tap.emitDiagnostic("Test 3:    I6  I6  I6  I6  I7  I7  I7  I7");
		
		ExperimentStatus<8> status;
		ExpectedOutcome<8> expected;
		expected._taskAWaitsTaskB[5][0] = true; expected._taskAWaitsTaskB[5][3] = true;
		expected._taskAWaitsTaskB[7][5] = true;
		
		#pragma oss task  weakin(var[0;8])    label(_i0 _i0 _i0 _i0 _i0 _i0 _i0 _i0)  shared(status, expected)
		{
			taskPrologue(0, status, expected);
			
			#pragma oss task  in(var[2;4])     label(___ ___ _I1 _I1 _I1 _I1 ___ ___)  shared(status, expected)
			taskCode(1, status, expected, 0);
			
			taskEpilogue(0, status, expected);
		}
		
		#pragma oss task weakinout(var[0;8])  label(io2 io2 io2 io2 io2 io2 io2 io2)  shared(status, expected)
		{
			taskPrologue(2, status, expected);
			
			#pragma oss task in(var[0;8])      label(_I3 _I3 _I3 _I3 _I3 _I3 _I3 _I3)  shared(status, expected)
			taskCode(3, status, expected, 2);
			
			#pragma oss task in(var[0;6])      label(_I4 _I4 _I4 _I4 _I4 _I4 ___ ___)  shared(status, expected)
			taskCode(4, status, expected, 2);
			
			#pragma oss task out(var[6;2])     label(___ ___ ___ ___ ___ ___ _O5 _O5)  shared(status, expected)
			taskCode(5, status, expected, 2);
			
			taskEpilogue(2, status, expected);
		}
		
		#pragma oss task  in(var[0;4])       label(_I6 _I6 _I6 _I6 ___ ___ ___ ___)  shared(status, expected)
		taskCode(6, status, expected);
		
		#pragma oss task  in(var[4;4])       label(___ ___ ___ ___ _I7 _I7 _I7 _I7)  shared(status, expected)
		taskCode(7, status, expected);
		
		#pragma oss taskwait
	}
	
	tap.end();
	
	return 0;
}

