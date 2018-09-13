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
	
	// Number of lower numbered tasks, concurrent to this one, that have started
	Atomic<int> _taskStartedLowerNumberedConcurrentTasks[NUM_TASKS];
	
	// Number of higher numbered tasks, concurrent to this one, that have finished
	Atomic<int> _taskFinishedHigherNumberedConcurrentTasks[NUM_TASKS];
	
	ExperimentStatus()
		: _taskHasStarted(), _taskHasFinished()
	{
		for (int i=0; i < NUM_TASKS; i++) {
			_taskHasStarted[i].store(false);
			_taskHasFinished[i].store(false);
			_taskStartedLowerNumberedConcurrentTasks[i].store(0);
			_taskFinishedHigherNumberedConcurrentTasks[i].store(0);
		}
	}
};


template <int NUM_TASKS>
struct ExpectedOutcome {
	bool _taskAWaitsTaskB[NUM_TASKS][NUM_TASKS];
	bool _taskARunsConcurrentlyToTaskB[NUM_TASKS][NUM_TASKS];
	
	ExpectedOutcome()
		: _taskAWaitsTaskB()
	{
		for (int i=0; i < NUM_TASKS; i++) {
			for (int j=0; j < NUM_TASKS; j++) {
				_taskAWaitsTaskB[i][j] = false;
				_taskARunsConcurrentlyToTaskB[i][j] = false;
			}
		}
	}
};


template <int NUM_TASKS>
static inline void taskCode(int currentTaskNumber, ExperimentStatus<NUM_TASKS> &status, ExpectedOutcome<NUM_TASKS> &expected)
{
	status._taskHasStarted[currentTaskNumber] = true;
	
	// Verify predecessors
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
	
	int totalLowerNumberedConcurrentTasks = 0;
	int totalHigherNumberedConcurrentTasks = 0;
	
	// Notify to all higher numbered concurrents that this one has started
	for (int otherTaskNumber = 0; otherTaskNumber < NUM_TASKS; otherTaskNumber++) {
		if (expected._taskARunsConcurrentlyToTaskB[currentTaskNumber][otherTaskNumber]) {
			if (otherTaskNumber < currentTaskNumber) {
				totalLowerNumberedConcurrentTasks++;
			} else if (currentTaskNumber < otherTaskNumber) {
				status._taskStartedLowerNumberedConcurrentTasks[otherTaskNumber]++;
				totalHigherNumberedConcurrentTasks++;
			}
		}
	}
	
	// Wait for all lower numbered concurrents to start
	// [concurrent check 1]
	{
		std::ostringstream oss;
		
		oss << "Evaluating that when T" << currentTaskNumber << " starts, also " << totalLowerNumberedConcurrentTasks << " concurrent and lower numbered tasks can start";
		
		
		tap.timedEvaluate(
			Equal< Atomic<int>, int >(status._taskStartedLowerNumberedConcurrentTasks[currentTaskNumber], totalLowerNumberedConcurrentTasks),
			SUSTAIN_MICROSECONDS,
			oss.str()
		);
	}
	
	// Wait for all higher numbered concurrents to finish
	// [concurrent check 2]
	{
		std::ostringstream oss;
		
		oss << "Evaluating that when T" << currentTaskNumber << " runs, also " << totalHigherNumberedConcurrentTasks << " concurrent and higher numbered tasks can finish";
		
		tap.timedEvaluate(
			Equal< Atomic<int>, int >(status._taskFinishedHigherNumberedConcurrentTasks[currentTaskNumber], totalHigherNumberedConcurrentTasks),
			SUSTAIN_MICROSECONDS,
			oss.str()
		);
	}
	
	status._taskHasFinished[currentTaskNumber] = true;
	
	// Notify to all lower numbered concurrents that this one has finished
	for (int otherTaskNumber = 0; otherTaskNumber < currentTaskNumber; otherTaskNumber++) {
		if (expected._taskARunsConcurrentlyToTaskB[currentTaskNumber][otherTaskNumber]) {
			status._taskFinishedHigherNumberedConcurrentTasks[otherTaskNumber]++;
		}
	}
	
	// Verify successors
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


static inline int numSubtests(int numWaits, int numConcurrents)
{
	return numWaits * 2 + (numConcurrents * numConcurrents) * 2;
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
		numSubtests(1, 0)
		+ numSubtests(3, 0)
		+ numSubtests(1, 2)
	);
	
	tap.begin();
	
	// Test 1
	{
		tap.emitDiagnostic("Test 1:   An if(0)'ed task causes its parent to stop until the task finishes");
		
		ExperimentStatus<2> status;
		ExpectedOutcome<2> expected;
		expected._taskAWaitsTaskB[0][1] = true;
		
		#pragma oss task shared(status, expected) if(0) label(T1: if 0 task)
		{
			taskCode(1, status, expected);
		}
		taskCode(0, status, expected);
	}
	
	
	// Test 2
	{
		tap.emitDiagnostic("Test 2:   if0'ed tasks respect their own dependencies");
		
		ExperimentStatus<3> status;
		ExpectedOutcome<3> expected;
		expected._taskAWaitsTaskB[2][1] = true;
		expected._taskAWaitsTaskB[0][1] = true;
		expected._taskAWaitsTaskB[0][2] = true;
		int a;
		
		#pragma oss task inout(a) shared(status, expected) label(T1: if 0 predecessor)
		{
			taskCode(1, status, expected);
		}
		
		#pragma oss task inout(a) shared(status, expected) if(0) label(T2: if 0 task)
		{
			taskCode(2, status, expected);
		}
		taskCode(0, status, expected);
	}
	
	
	// Test 3
	{
		tap.emitDiagnostic("Test 3:   children of if0'ed tasks do not delay their grandparent");
		
		ExperimentStatus<3> status;
		ExpectedOutcome<3> expected;
		expected._taskAWaitsTaskB[0][1] = true;
		expected._taskARunsConcurrentlyToTaskB[0][2] = true;
		expected._taskARunsConcurrentlyToTaskB[2][0] = true;
		int a;
		
		#pragma oss task shared(status, expected) label(T1: if 0 task)
		{
			#pragma oss task shared(status, expected) label(T1.1: child of if 0 task)
			{
				taskCode(2, status, expected);
			}
			
			taskCode(1, status, expected);
		}
		
		taskCode(0, status, expected);
		
		#pragma oss taskwait
	}
	
	
	tap.end();
	
	return 0;
}

