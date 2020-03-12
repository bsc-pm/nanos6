/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
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

	tap.registerNewTests(6*2 + 2*2 + 3*2 + 3*2 + 3*2);
	tap.begin();

	int var;
	int var2;
	int var3[10];

	// Test 1
	{
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[1][0] = true;
		expected._taskAWaitsTaskB[2][0] = true; expected._taskAWaitsTaskB[2][1] = true;
		expected._taskAWaitsTaskB[3][0] = true; expected._taskAWaitsTaskB[3][1] = true; expected._taskAWaitsTaskB[3][2] = true;

		#pragma oss task inout(var) label(O0)  shared(status, expected)
		taskCode(0, status, expected);

		#pragma oss task inout(var) label(O1)  shared(status, expected)
		taskCode(1, status, expected);

		#pragma oss task inout(var) label(O2)  shared(status, expected)
		taskCode(2, status, expected);

		#pragma oss task inout(var) label(O3)  shared(status, expected)
		taskCode(3, status, expected);

		#pragma oss taskwait
	}

	// Test 2
	{
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[2][0] = true; expected._taskAWaitsTaskB[2][1] = true;

		#pragma oss task in(var) label(O0)  shared(status, expected)
		taskCode(0, status, expected);

		#pragma oss task in(var) label(O1)  shared(status, expected)
		taskCode(1, status, expected);

		#pragma oss task out(var) label(O2)  shared(status, expected)
		taskCode(2, status, expected);

		#pragma oss task in(var2) label(O3)  shared(status, expected)
		taskCode(3, status, expected);

		#pragma oss taskwait
	}

	// Test 3
	{
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[1][0] = true;
		expected._taskAWaitsTaskB[2][0] = true; expected._taskAWaitsTaskB[2][1] = true;

		#pragma oss task inout(var3[0;10]) label(O0)  shared(status, expected)
		taskCode(0, status, expected);

		#pragma oss task in(var3[0]) label(O1)  shared(status, expected)
		taskCode(1, status, expected);

		#pragma oss task out(var3[0]) label(O2)  shared(status, expected)
		taskCode(2, status, expected);

		#pragma oss task in(var3[1]) label(O3)  shared(status, expected)
		taskCode(3, status, expected);

		#pragma oss taskwait
	}

	// Test 4
	{
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[3][0] = true; expected._taskAWaitsTaskB[3][1] = true; expected._taskAWaitsTaskB[3][2] = true;

		#pragma oss task out(var) label(O0)  shared(status, expected)
		taskCode(0, status, expected);

		#pragma oss task reduction(+: var) label(O1)  shared(status, expected)
		taskCode(1, status, expected);

		#pragma oss task reduction(+: var) label(O2)  shared(status, expected)
		taskCode(2, status, expected);

		#pragma oss task in(var) label(O3)  shared(status, expected)
		taskCode(3, status, expected);

		#pragma oss taskwait
	}

	// Test 5
	{
		ExperimentStatus<4> status;
		ExpectedOutcome<4> expected;
		expected._taskAWaitsTaskB[3][0] = true; expected._taskAWaitsTaskB[3][1] = true; expected._taskAWaitsTaskB[3][2] = true;

		#pragma oss task reduction(+: [10]var3) label(O0)  shared(status, expected)
		taskCode(0, status, expected);

		#pragma oss task reduction(+: [10]var3) label(O1)  shared(status, expected)
		taskCode(1, status, expected);

		#pragma oss task reduction(+: [10]var3) label(O2)  shared(status, expected)
		taskCode(2, status, expected);

		#pragma oss task in(var3[0]) label(O3)  shared(status, expected)
		taskCode(3, status, expected);

		#pragma oss taskwait
	}

	tap.end();

	return 0;
}

