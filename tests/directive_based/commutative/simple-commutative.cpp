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
	bool _taskACannotRunConcurrentlyWithB[NUM_TASKS][NUM_TASKS];
	
	ExpectedOutcome()
		: _taskAWaitsTaskB()
	{
		for (int i=0; i < NUM_TASKS; i++) {
			for (int j=0; j < NUM_TASKS; j++) {
				_taskAWaitsTaskB[i][j] = false;
				_taskACannotRunConcurrentlyWithB[i][j] = false;
			}
		}
	}
};


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
	
	for (int otherTaskNumber = currentTaskNumber+1; otherTaskNumber < NUM_TASKS; otherTaskNumber++) {
		std::ostringstream oss;
		
		if (expected._taskACannotRunConcurrentlyWithB[currentTaskNumber][otherTaskNumber]) {
			oss << "Evaluating that T" << otherTaskNumber << " does not run concurrently with T" << currentTaskNumber;
			
			bool otherStarted = status._taskHasStarted[otherTaskNumber].load();
			bool otherFinished = status._taskHasFinished[otherTaskNumber].load();
			
			Equal<bool, bool> cond1(otherStarted, otherFinished);
			Equal<bool, Atomic<bool> > cond2(otherStarted, status._taskHasStarted[otherTaskNumber]);
			Equal<bool, Atomic<bool> > cond3(otherFinished, status._taskHasFinished[otherTaskNumber]);
			And< Equal<bool, bool>, Equal<bool, Atomic<bool> > > cond12(cond1, cond2);
			And<  And< Equal<bool, bool>, Equal<bool, Atomic<bool> > >,  Equal<bool, Atomic<bool> >  > cond123(cond12, cond3);
			
			tap.sustainedEvaluate(
				cond123,
				SUSTAIN_MICROSECONDS,
				oss.str()
			);
		}
	}
	
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


static inline int numSubtests(int numTasks, int numWaits, int numExclussions)
{
	#if PERFORM_NON_GUARANTEED_CHECKS
		return numTasks * (numTasks - 1) + numExclussions;
	#else
		return numWaits * 2 + numExclussions;
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
		numSubtests(9, 8, 6) +
		numSubtests(9, 0, 20) +
		numSubtests(9, 0, 20)
	);
	
	tap.begin();
	
	int var[8];
	
	
	// Test 1
	{
		tap.emitDiagnostic("Test 1:   VAR0 VAR1 VAR2 VAR3 VAR4 VAR5");
		tap.emitDiagnostic("Test 1:   -----------------------------");
		tap.emitDiagnostic("Test 1:   IO0                 IO0      ");
		tap.emitDiagnostic("Test 1:   IO1            IO1           ");
		tap.emitDiagnostic("Test 1:   IO2       IO2                ");
		tap.emitDiagnostic("Test 1:   IO3  IO3                     ");
		tap.emitDiagnostic("Test 1:   IO4                          ");
		tap.emitDiagnostic("Test 1:        IO5                 CM5 ");
		tap.emitDiagnostic("Test 1:             IO6            CM6 ");
		tap.emitDiagnostic("Test 1:                  IO7       CM7 ");
		tap.emitDiagnostic("Test 1:                       IO8  CM8 ");
		
		ExperimentStatus<9> status;
		ExpectedOutcome<9> expected;
		
		expected._taskAWaitsTaskB[1][0] = true;
		expected._taskAWaitsTaskB[2][1] = true;
		expected._taskAWaitsTaskB[3][2] = true;
		expected._taskAWaitsTaskB[4][3] = true;
		expected._taskAWaitsTaskB[5][3] = true;
		expected._taskAWaitsTaskB[6][2] = true;
		expected._taskAWaitsTaskB[7][1] = true;
		expected._taskAWaitsTaskB[8][0] = true;
		expected._taskACannotRunConcurrentlyWithB[5][6] = true;
		expected._taskACannotRunConcurrentlyWithB[5][7] = true;
		expected._taskACannotRunConcurrentlyWithB[5][8] = true;
		expected._taskACannotRunConcurrentlyWithB[6][7] = true;
		expected._taskACannotRunConcurrentlyWithB[6][8] = true;
		expected._taskACannotRunConcurrentlyWithB[7][8] = true;
		
		#pragma oss task inout(var[0]) inout(var[4]) label(IO0                 IO0      )  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task inout(var[0]) inout(var[3]) label(IO1            IO1           )  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task inout(var[0]) inout(var[2]) label(IO2       IO2                )  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task inout(var[0]) inout(var[1]) label(IO3  IO3                     )  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss task inout(var[0]) label(IO4                          )  shared(status, expected)
		taskCode(4, status, expected);
		
		#pragma oss task inout(var[1]) commutative(var[5]) label(     IO5                 CM5 )  shared(status, expected)
		taskCode(5, status, expected);
		
		#pragma oss task inout(var[2]) commutative(var[5]) label(          IO6            CM6 )  shared(status, expected)
		taskCode(6, status, expected);
		
		#pragma oss task inout(var[3]) commutative(var[5]) label(               IO7       CM7 )  shared(status, expected)
		taskCode(7, status, expected);
		
		#pragma oss task inout(var[4]) commutative(var[5]) label(                    IO8  CM8 )  shared(status, expected)
		taskCode(8, status, expected);
		
		#pragma oss taskwait
	}
	
	// Test 2
	{
		tap.emitDiagnostic("Test 2:   VAR0 VAR1 VAR2 VAR3 VAR4 VAR5");
		tap.emitDiagnostic("Test 2:   -----------------------------");
		tap.emitDiagnostic("Test 2:   CM0                 CM0      ");
		tap.emitDiagnostic("Test 2:   CM1            CM1           ");
		tap.emitDiagnostic("Test 2:   CM2       CM2                ");
		tap.emitDiagnostic("Test 2:   CM3  CM3                     ");
		tap.emitDiagnostic("Test 2:   CM4                          ");
		tap.emitDiagnostic("Test 2:        CM5                 CM5 ");
		tap.emitDiagnostic("Test 2:             CM6            CM6 ");
		tap.emitDiagnostic("Test 2:                  CM7       CM7 ");
		tap.emitDiagnostic("Test 2:                       CM8  CM8 ");
		
		ExperimentStatus<9> status;
		ExpectedOutcome<9> expected;
		
		expected._taskACannotRunConcurrentlyWithB[0][1] = true;
		expected._taskACannotRunConcurrentlyWithB[0][2] = true;
		expected._taskACannotRunConcurrentlyWithB[0][3] = true;
		expected._taskACannotRunConcurrentlyWithB[0][4] = true;
		expected._taskACannotRunConcurrentlyWithB[1][2] = true;
		expected._taskACannotRunConcurrentlyWithB[1][3] = true;
		expected._taskACannotRunConcurrentlyWithB[1][4] = true;
		expected._taskACannotRunConcurrentlyWithB[2][3] = true;
		expected._taskACannotRunConcurrentlyWithB[2][4] = true;
		expected._taskACannotRunConcurrentlyWithB[3][4] = true;
		
		expected._taskACannotRunConcurrentlyWithB[3][5] = true;
		expected._taskACannotRunConcurrentlyWithB[2][6] = true;
		expected._taskACannotRunConcurrentlyWithB[1][7] = true;
		expected._taskACannotRunConcurrentlyWithB[0][8] = true;
		
		expected._taskACannotRunConcurrentlyWithB[5][6] = true;
		expected._taskACannotRunConcurrentlyWithB[5][7] = true;
		expected._taskACannotRunConcurrentlyWithB[5][8] = true;
		expected._taskACannotRunConcurrentlyWithB[6][7] = true;
		expected._taskACannotRunConcurrentlyWithB[6][8] = true;
		expected._taskACannotRunConcurrentlyWithB[7][8] = true;
		
		#pragma oss task commutative(var[0], var[4]) label(CM0                 CM0      )  shared(status, expected)
		taskCode(0, status, expected);
		
		#pragma oss task commutative(var[0], var[3]) label(CM1            CM1           )  shared(status, expected)
		taskCode(1, status, expected);
		
		#pragma oss task commutative(var[0], var[2]) label(CM2       CM2                )  shared(status, expected)
		taskCode(2, status, expected);
		
		#pragma oss task commutative(var[0], var[1]) label(CM3  CM3                     )  shared(status, expected)
		taskCode(3, status, expected);
		
		#pragma oss task commutative(var[0]) label(CM4                          )  shared(status, expected)
		taskCode(4, status, expected);
		
		#pragma oss task commutative(var[1], var[5]) label(     CM5                 CM5 )  shared(status, expected)
		taskCode(5, status, expected);
		
		#pragma oss task commutative(var[2], var[5]) label(          CM6            CM6 )  shared(status, expected)
		taskCode(6, status, expected);
		
		#pragma oss task commutative(var[3], var[5]) label(               CM7       CM7 )  shared(status, expected)
		taskCode(7, status, expected);
		
		#pragma oss task commutative(var[4], var[5]) label(                    CM8  CM8 )  shared(status, expected)
		taskCode(8, status, expected);
		
		#pragma oss taskwait
	}
	
	// Test 2
	{
		tap.emitDiagnostic("Test 2:   VAR0  VAR1  VAR2  VAR3  VAR4  VAR5 ");
		tap.emitDiagnostic("Test 2:  ------------------------------------");
		tap.emitDiagnostic("Test 2:   cm0   cm0   cm0   cm0   cm0        ");
		tap.emitDiagnostic("Test 2:   CM0.0                   CM0.0      ");
		tap.emitDiagnostic("Test 2:   CM0.1             CM0.1            ");
		tap.emitDiagnostic("Test 2:   cm1   cm1   cm1                    ");
		tap.emitDiagnostic("Test 2:   CM1.2       CM1.2                  ");
		tap.emitDiagnostic("Test 2:   CM1.3 CM1.3                        ");
		tap.emitDiagnostic("Test 2:   CM2_4_                             ");
		tap.emitDiagnostic("Test 2:        cm3  cm3           cm3        ");
		tap.emitDiagnostic("Test 2:        CM3.5              CM3.5      ");
		tap.emitDiagnostic("Test 2:             CM3.6         CM3.6      ");
		tap.emitDiagnostic("Test 2:                     cm4   cm4   cm4  ");
		tap.emitDiagnostic("Test 2:                     CM4.7       CM4.7");
		tap.emitDiagnostic("Test 2:                           CM4.8 CM4.8");
		
		ExperimentStatus<9> status;
		ExpectedOutcome<9> expected;
		
		expected._taskACannotRunConcurrentlyWithB[0][1] = true;
		expected._taskACannotRunConcurrentlyWithB[0][2] = true;
		expected._taskACannotRunConcurrentlyWithB[0][3] = true;
		expected._taskACannotRunConcurrentlyWithB[0][4] = true;
		expected._taskACannotRunConcurrentlyWithB[1][2] = true;
		expected._taskACannotRunConcurrentlyWithB[1][3] = true;
		expected._taskACannotRunConcurrentlyWithB[1][4] = true;
		expected._taskACannotRunConcurrentlyWithB[2][3] = true;
		expected._taskACannotRunConcurrentlyWithB[2][4] = true;
		expected._taskACannotRunConcurrentlyWithB[3][4] = true;
		
		expected._taskACannotRunConcurrentlyWithB[3][5] = true;
		expected._taskACannotRunConcurrentlyWithB[2][6] = true;
		expected._taskACannotRunConcurrentlyWithB[1][7] = true;
		expected._taskACannotRunConcurrentlyWithB[0][8] = true;
		
		expected._taskACannotRunConcurrentlyWithB[5][6] = true;
		expected._taskACannotRunConcurrentlyWithB[5][7] = true;
		expected._taskACannotRunConcurrentlyWithB[5][8] = true;
		expected._taskACannotRunConcurrentlyWithB[6][7] = true;
		expected._taskACannotRunConcurrentlyWithB[6][8] = true;
		expected._taskACannotRunConcurrentlyWithB[7][8] = true;
		
		#pragma oss task weakcommutative(var[0:4]) label(cm0   cm0   cm0   cm0   cm0        ) shared(status, expected)
		{
			#pragma oss task commutative(var[0], var[4]) label(CM0.0                   CM0.0      )  shared(status, expected)
			taskCode(0, status, expected);
			
			#pragma oss task commutative(var[0], var[3]) label(CM0.1             CM0.1            )  shared(status, expected)
			taskCode(1, status, expected);
		}
		
		#pragma oss task weakcommutative(var[0:2]) label(cm1   cm1   cm1                    ) shared(status, expected)
		{
			#pragma oss task commutative(var[0], var[2]) label(CM1.2       CM1.2                  )  shared(status, expected)
			taskCode(2, status, expected);
			
			#pragma oss task commutative(var[0], var[1]) label(CM1.3 CM1.3                        )  shared(status, expected)
			taskCode(3, status, expected);
		}
		
		#pragma oss task commutative(var[0]) label(CM2_4_                             )  shared(status, expected)
		taskCode(4, status, expected);
		
		#pragma oss task weakcommutative(var[1:2], var[4]) label(     cm3  cm3           cm3        ) shared(status, expected)
		{
			#pragma oss task commutative(var[1], var[5]) label(     CM3.5              CM3.5      )  shared(status, expected)
			taskCode(5, status, expected);
			
			#pragma oss task commutative(var[2], var[5]) label(          CM3.6         CM3.6      )  shared(status, expected)
			taskCode(6, status, expected);
		}
		
		#pragma oss task weakcommutative(var[3:5]) label(                  cm4   cm4   cm4  ) shared(status, expected)
		{
			#pragma oss task commutative(var[3], var[5]) label(                  CM4.7       CM4.7)  shared(status, expected)
			taskCode(7, status, expected);
			
			#pragma oss task commutative(var[4], var[5]) label(                        CM4.8 CM4.8)  shared(status, expected)
			taskCode(8, status, expected);
		}
		
		#pragma oss taskwait
	}
	
	tap.end();
	
	return 0;
}

