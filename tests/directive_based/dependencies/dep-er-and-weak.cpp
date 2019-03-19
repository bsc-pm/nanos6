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


#include <Atomic.hpp>
#include <Functors.hpp>
#define SUSTAIN_MICROSECONDS 100000L


using namespace Functors;


TestAnyProtocolProducer tap;


struct TaskInformation {
	std::string _label;
	Atomic<bool> _hasStarted;
	Atomic<bool> _hasFinished;
	
	TaskInformation()
		: _hasStarted(false), _hasFinished(false)
	{
	}
};


// _finishesBefore contains the task id of a task that must forcibly finish before the task identified by _startsAfter starts
struct StrictOrder {
	int _finishesBefore;
	int _startsAfter;
};


// _canEndAfterStarting contains a task id of a task that may end after starting (and possibly ending) the task identified by _canStartBeforeEnding
// _canStartBeforeEnding contains a task id of a task that may start before ending the task identified by _canEndAfterStarting
struct RelaxedOrder {
	int _canEndAfterStarting;
	int _canStartBeforeEnding;
};


static void verifyStrictOrder(StrictOrder const &constraint, TaskInformation *taskInformation, bool isSource)
{
	if (isSource) {
		std::ostringstream oss;
		
		oss
			<< "Evaluating that " << taskInformation[constraint._startsAfter]._label
			<< " does not start before "  << taskInformation[constraint._finishesBefore]._label
			<< " finishes";
		
		tap.sustainedEvaluate(
			False< Atomic<bool> >(taskInformation[constraint._startsAfter]._hasStarted),
			SUSTAIN_MICROSECONDS,
			oss.str()
		);
	} else {
		std::ostringstream oss;
		
		oss
		<< "Evaluating that " << taskInformation[constraint._startsAfter]._label
		<< " starts after "  << taskInformation[constraint._finishesBefore]._label
		<< " has finished";
		
		tap.evaluate(taskInformation[constraint._finishesBefore]._hasFinished.load(), oss.str());
	}
}


// NOTE: The target must call this function before marking itself as started
static void verifyRelaxedOrder(RelaxedOrder const &constraint, TaskInformation *taskInformation, bool isSource)
{
	if (isSource) {
		std::ostringstream oss;
		
		oss
		<< "Evaluating that when " << taskInformation[constraint._canEndAfterStarting]._label
		<< " finishes, "  << taskInformation[constraint._canStartBeforeEnding]._label
		<< " has started";
		
		tap.timedEvaluate(
			True< Atomic<bool> >(taskInformation[constraint._canStartBeforeEnding]._hasStarted),
			2 * SUSTAIN_MICROSECONDS,
			oss.str()
		);
	} else {
		assert(!taskInformation[constraint._canStartBeforeEnding]._hasStarted.load());
		
		std::ostringstream oss;
		
		oss
		<< "Evaluating that when " << taskInformation[constraint._canStartBeforeEnding]._label
		<< " starts, "  << taskInformation[constraint._canEndAfterStarting]._label
		<< " has not finished";
		
		tap.evaluate(
			!taskInformation[constraint._canEndAfterStarting]._hasFinished.load(),
			oss.str()
		);
	}
}





enum Tasks {
	T1=0,
	T1_1,
	T1_2,
	T2,
	T2_1,
	T2_1_1,
	T2_2,
	T2_3,
	T3,
	NUM_TASKS
};


int main(int argc, char **argv)
{
	nanos6_wait_for_full_initialization();
	
	long activeCPUs = nanos6_get_num_cpus();
	if (activeCPUs <= 2) {
		// This test only works correctly with more than 2 CPUs
		tap.registerNewTests(1);
		tap.begin();
		tap.skip("This test requires more than 2 CPUs");
		tap.end();
		return 0;
	}
	
	tap.registerNewTests(3*2 + 6*2 + 7*2);
	tap.begin();
	
	
	// Test 1
	// WIN {IN} -- WOUT {OUT}
	{
		tap.emitDiagnostic("Test 1:   WIN {IN} -- WOUT {OUT}");
// 		enum Tasks {
// 			T1=0,
// 			T1_1,
// 			T2,
// 			T2_1,
// 			NUM_TASKS
// 		};
		
		TaskInformation taskInformation[NUM_TASKS];
		taskInformation[T1]._label = "T1";
		taskInformation[T1_1]._label = "T1_1";
		taskInformation[T2]._label = "T2";
		taskInformation[T2_1]._label = "T2_1";
		
		RelaxedOrder r1_2 = {T1, T2};
		RelaxedOrder r11_2 = {T1_1, T2};
		StrictOrder s11_22 = {T1_1, T2_1};
		
		int var1;
		
		#pragma oss task shared(var1, taskInformation) weakin(var1) label(T1 WIN)
		{
			taskInformation[T1]._hasStarted = true;
			#pragma oss task shared(var1, taskInformation) in(var1) label(T1_1 IN)
			{
				taskInformation[T1_1]._hasStarted = true;
				verifyRelaxedOrder(r11_2, taskInformation, true);
				verifyStrictOrder(s11_22, taskInformation, true);
				taskInformation[T1_1]._hasFinished = true;
			}
			
			verifyRelaxedOrder(r1_2, taskInformation, true);
			taskInformation[T1]._hasFinished = true;
		}
		
		#pragma oss task shared(var1, taskInformation) weakout(var1) label(T2 WOUT)
		{
			verifyRelaxedOrder(r1_2, taskInformation, false);
			verifyRelaxedOrder(r11_2, taskInformation, false);
			taskInformation[T2]._hasStarted = true;
			
			#pragma oss task shared(var1, taskInformation) out(var1) label(T2_1 OUT)
			{
				taskInformation[T2_1]._hasStarted = true;
				verifyStrictOrder(s11_22, taskInformation, false);
				taskInformation[T2_1]._hasFinished = true;
			}
			taskInformation[T2]._hasFinished = true;
		}
		
		#pragma oss taskwait
	}
	
	
	// Test 2
	// WINOUT {INOUT -- IN} -- WIN {IN}
	{
		tap.emitDiagnostic("Test 2:   WINOUT {INOUT -- IN} -- WIN {IN}");
		TaskInformation taskInformation[NUM_TASKS];
		taskInformation[T1]._label = "T1";
		taskInformation[T1_1]._label = "T1_1";
		taskInformation[T1_2]._label = "T1_2";
		taskInformation[T2]._label = "T2";
		taskInformation[T2_1]._label = "T2_1";
		
		RelaxedOrder r1_2 = {T1, T2};
		RelaxedOrder r11_2 = {T1_1, T2};
		RelaxedOrder r12_2 = {T1_2, T2};
		StrictOrder s11_12 = {T1_1, T1_2};
		StrictOrder s1_21 = {T1, T2_1};
		StrictOrder s11_21 = {T1_1, T2_1};
		
		int var1;
		
		#pragma oss task shared(var1, taskInformation) weakinout(var1) label(T1 WINOUT)
		{
			taskInformation[T1]._hasStarted = true;
			#pragma oss task shared(var1, taskInformation) inout(var1) label(T1_1 INOUT)
			{
				taskInformation[T1_1]._hasStarted = true;
				verifyRelaxedOrder(r11_2, taskInformation, true);
				verifyStrictOrder(s11_12, taskInformation, true);
				verifyStrictOrder(s11_21, taskInformation, true);
				taskInformation[T1_1]._hasFinished = true;
			}
			
			#pragma oss task shared(var1, taskInformation) in(var1) label(T1_2 IN)
			{
				taskInformation[T1_2]._hasStarted = true;
				verifyStrictOrder(s11_12, taskInformation, false);
				verifyRelaxedOrder(r12_2, taskInformation, true);
				taskInformation[T1_2]._hasFinished = true;
			}
			
			verifyRelaxedOrder(r1_2, taskInformation, true);
			verifyStrictOrder(s1_21, taskInformation, true);
			taskInformation[T1]._hasFinished = true;
		}
		
		#pragma oss task shared(var1, taskInformation) weakin(var1) label(T2 WIN)
		{
			verifyRelaxedOrder(r1_2, taskInformation, false);
			verifyRelaxedOrder(r11_2, taskInformation, false);
			verifyRelaxedOrder(r12_2, taskInformation, false);
			taskInformation[T2]._hasStarted = true;
			
			#pragma oss task shared(var1, taskInformation) in(var1) label(T2_1 IN)
			{
				taskInformation[T2_1]._hasStarted = true;
				verifyStrictOrder(s1_21, taskInformation, false);
				verifyStrictOrder(s11_21, taskInformation, false);
				taskInformation[T2_1]._hasFinished = true;
			}
			taskInformation[T2]._hasFinished = true;
		}
		
		#pragma oss taskwait
	}
	
	
	// Test 3
	// WIN {IN} -- WINOUT {WIN {IN} -- OUT -- IN} -- IN
	{
		tap.emitDiagnostic("Test 3:   WIN {IN} -- WINOUT {WIN {IN} -- OUT -- IN} -- IN");
		
		TaskInformation taskInformation[NUM_TASKS];
		taskInformation[T1]._label = "T1";
		taskInformation[T1_1]._label = "T1_1";
		taskInformation[T2]._label = "T2";
		taskInformation[T2_1]._label = "T2_1";
		taskInformation[T2_1_1]._label = "T2_1_1";
		taskInformation[T2_2]._label = "T2_2";
		taskInformation[T2_3]._label = "T2_3";
		taskInformation[T3]._label = "T3";
		
		RelaxedOrder r1_2 = {T1, T2};
		RelaxedOrder r1_21 = {T1, T2_1};
		StrictOrder s2_3 = {T2, T3};
		StrictOrder s22_3 = {T2_2, T3};
		StrictOrder s11_22 = {T1_1, T2_2};
		StrictOrder s211_22 = {T2_1_1, T2_2};
		StrictOrder s22_23 = {T2_2, T2_3};
		
		int var1;
		
		#pragma oss task shared(var1, taskInformation) weakin(var1) label(T1 WIN)
		{
			taskInformation[T1]._hasStarted = true;
			#pragma oss task shared(var1, taskInformation) in(var1) label(T1_1 IN)
			{
				taskInformation[T1_1]._hasStarted = true;
				verifyStrictOrder(s11_22, taskInformation, true);
				taskInformation[T1_1]._hasFinished = true;
			}
			
			verifyRelaxedOrder(r1_2, taskInformation, true);
			verifyRelaxedOrder(r1_21, taskInformation, true);
			taskInformation[T1]._hasFinished = true;
		}
		
		#pragma oss task shared(var1, taskInformation) weakinout(var1) label(T2 WINOUT)
		{
			verifyRelaxedOrder(r1_2, taskInformation, false);
			taskInformation[T2]._hasStarted = true;
			
			#pragma oss task shared(var1, taskInformation) weakin(var1) label(T2_1 WIN)
			{
				verifyRelaxedOrder(r1_21, taskInformation, false);
				taskInformation[T2_1]._hasStarted = true;
				
				#pragma oss task shared(var1, taskInformation) in(var1) label(T2_1_1 IN)
				{
					taskInformation[T2_1_1]._hasStarted = true;
					verifyStrictOrder(s211_22, taskInformation, true);
					taskInformation[T2_1_1]._hasFinished = true;
				}
				
				taskInformation[T2_1]._hasFinished = true;
			}
			
			#pragma oss task shared(var1, taskInformation) out(var1) label(T2_2 OUT)
			{
				taskInformation[T2_2]._hasStarted = true;
				verifyStrictOrder(s11_22, taskInformation, false);
				verifyStrictOrder(s211_22, taskInformation, false);
				verifyStrictOrder(s22_23, taskInformation, true);
				verifyStrictOrder(s22_3, taskInformation, true);
				taskInformation[T2_2]._hasFinished = true;
			}
			
			#pragma oss task shared(var1, taskInformation) in(var1) label(T2_3 IN)
			{
				taskInformation[T2_3]._hasStarted = true;
				verifyStrictOrder(s22_23, taskInformation, false);
				taskInformation[T2_3]._hasFinished = true;
			}
			
			verifyStrictOrder(s2_3, taskInformation, true);
			
			taskInformation[T2]._hasFinished = true;
		}
		
		#pragma oss task shared(var1, taskInformation) in(var1) label(T3 IN)
		{
			taskInformation[T3]._hasStarted = true;
			verifyStrictOrder(s2_3, taskInformation, false);
			verifyStrictOrder(s22_3, taskInformation, false);
			taskInformation[T3]._hasFinished = true;
		}
		
		#pragma oss taskwait
	}
	
	tap.end();
	
	return 0;
}

