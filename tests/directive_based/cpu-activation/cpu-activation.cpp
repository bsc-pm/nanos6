/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>

#include <Atomic.hpp>
#include <Functors.hpp>
#include "TestAnyProtocolProducer.hpp"
#include "Timer.hpp"

#include "ConditionVariable.hpp"

#include <cassert>
#include <vector>

#include <sched.h>


#define VALIDATION_STEPS_PER_CPU 16


using namespace Functors;


TestAnyProtocolProducer tap;


static Atomic<long> _blockerCPU;

class Blocker {
public:
	ConditionVariable _condVar;
	
	Blocker()
	{
	}
	
	void body()
	{
		_blockerCPU = nanos6_get_current_system_cpu();
		_condVar.wait();
	}
};


class CPUStatusFunctor : public Functor {
	long _cpu;
	
public:
	typedef nanos6_cpu_status_t type;
	
	CPUStatusFunctor(long cpu)
		: _cpu(cpu)
	{
	}
	
	nanos6_cpu_status_t operator()()
	{
		return nanos6_get_cpu_status(_cpu);
	}
};


class PlacementEvaluator {
public:
	long _expectedCPU;
	std::vector< Atomic<int> > &_tasksPerVirtualCPU;
	
	PlacementEvaluator(long expectedCPU, std::vector< Atomic<int> > &tasksPerVirtualCPU)
		: _expectedCPU(expectedCPU), _tasksPerVirtualCPU(tasksPerVirtualCPU)
	{
	}
	
	void body()
	{
		long systemCPU = nanos6_get_current_system_cpu();
		long virtualCPU = nanos6_get_current_virtual_cpu();
		
		// Weak check since we cannot guarantee that a CPU will not run (only) one task
		tap.evaluateWeak(
			systemCPU == _expectedCPU,
			"Check that when only one CPU is enabled, all tasks run in that CPU",
			"Cannot guarantee that one task will not get past the status transition"
		);
		if (systemCPU != _expectedCPU) {
			tap.emitDiagnostic("Expected ", _expectedCPU, " got ", systemCPU);
			tap.emitDiagnostic("CPU ", _expectedCPU, " in activation status ", nanos6_get_cpu_status(_expectedCPU));
			tap.emitDiagnostic("CPU ", systemCPU, " in activation status ", nanos6_get_cpu_status(systemCPU));
		}
		
		_tasksPerVirtualCPU[virtualCPU]++;
	}
};


int main(int argc, char **argv) {
	nanos6_wait_for_full_initialization();
	
	long activeCPUs = nanos6_get_num_cpus();
	tap.emitDiagnostic("Detected ", activeCPUs, " CPUs");
	
	if (activeCPUs == 1) {
		// This test only works correctly with more than 1 CPU
		tap.registerNewTests(1);
		tap.begin();
		tap.skip("This test does not work with just 1 CPU");
		tap.end();
		return 0;
	}
	
	tap.registerNewTests(
		/* Phase 1 */ 6
		/* Phase 2 */ + activeCPUs*VALIDATION_STEPS_PER_CPU
		/* Phase 3 */ + (activeCPUs - 1)
	);
	tap.begin();
	
	
	Timer timer;
	
	
	/***********/
	/* PHASE 1 */
	/***********/
	
	tap.emitDiagnostic("*****************");
	tap.emitDiagnostic("***  PHASE 1  ***");
	tap.emitDiagnostic("***           ***");
	tap.emitDiagnostic("***  6 tests  ***");
	tap.emitDiagnostic("*****************");
	
	_blockerCPU = -1;
	Blocker *blocker = new Blocker();
	
	#pragma oss task label(blocker)
	blocker->body();
	
	// Wait to get the blocker CPU
	while (_blockerCPU == -1) {
		sched_yield();
	}
	
	long blockerCPU = _blockerCPU;
	
	tap.evaluate(
		nanos6_get_cpu_status(blockerCPU) == nanos6_enabled_cpu,
		"Check that the CPU that runs a task is enabled"
	); // 2
	
	nanos6_disable_cpu(blockerCPU);
	tap.evaluate(
		nanos6_get_cpu_status(blockerCPU) == nanos6_disabling_cpu,
		"Check that attempting to disable a CPU will set it to disabling status"
	); // 3
	tap.bailOutAndExitIfAnyFailed();
	
	CPUStatusFunctor cpuStatusFunctor(blockerCPU);
	nanos6_cpu_status_t expectedStatus = nanos6_disabled_cpu;
	
	blocker->_condVar.signal();
	tap.timedEvaluate(
		Equal<CPUStatusFunctor, nanos6_cpu_status_t>(cpuStatusFunctor, expectedStatus),
		1000000, // 1 second
		"Check that the CPU completes the deactivation in a reasonable amount of time"
	); // 4
	tap.bailOutAndExitIfAnyFailed();
	
	nanos6_disable_cpu(blockerCPU);
	tap.evaluate(
		nanos6_get_cpu_status(blockerCPU) == nanos6_disabled_cpu,
		"Check that attempting to disable an already disabled CPU keeps it untouched"
	); // 5
	
	expectedStatus = nanos6_enabled_cpu;
	nanos6_enable_cpu(blockerCPU);
	tap.timedEvaluate(
		Equal<CPUStatusFunctor, nanos6_cpu_status_t>(cpuStatusFunctor, expectedStatus),
		1000000, // 1 second
		"Check that enabling a CPU will eventually set it to enabled"
	); // 6
	tap.bailOutAndExitIfAnyFailed();
	
	nanos6_enable_cpu(blockerCPU);
	tap.evaluate(
		nanos6_get_cpu_status(blockerCPU) == nanos6_enabled_cpu,
		"Check that reenabling a CPU does not change its status"
	); // 7
	tap.bailOutAndExitIfAnyFailed();
	
	#pragma oss taskwait
	
	
	/***********/
	/* PHASE 2 */
	/***********/
	
	tap.emitDiagnostic("*****************");
	tap.emitDiagnostic("***  PHASE 2  ***");
	tap.emitDiagnostic("***           ***");
	tap.emitDiagnostic("***  ", activeCPUs*VALIDATION_STEPS_PER_CPU, " tests ***");
	tap.emitDiagnostic("*****************");
	
	long currentSystemCPU = nanos6_get_current_system_cpu();
	long currentVirtualCPU = nanos6_get_current_virtual_cpu();
	
	tap.emitDiagnostic("Will be using CPU ", currentSystemCPU);
	
	// Disable all other CPUs
	for (void *cpuIterator = nanos6_cpus_begin(); cpuIterator != nanos6_cpus_end(); cpuIterator = nanos6_cpus_advance(cpuIterator)) {
		long cpu = nanos6_cpus_get(cpuIterator);
		if (cpu != currentSystemCPU) {
			tap.emitDiagnostic("Disabling CPU ", cpu);
			nanos6_disable_cpu(cpu);
		} else {
			tap.emitDiagnostic("Not disabling CPU ", cpu);
		}
	}
	
	// Should be indexed with virtual CPU identifiers
	std::vector< Atomic<int> > tasksPerVirtualCPU(activeCPUs);
	for (int i=0; i < activeCPUs; i++) {
		tasksPerVirtualCPU[i] = 0;
	}
	
	for (int i=0; i < activeCPUs*VALIDATION_STEPS_PER_CPU; i++) {
		PlacementEvaluator *placementEvaluator = new PlacementEvaluator(currentSystemCPU, tasksPerVirtualCPU);
		
		#pragma oss task label(placement_evaluator)
		placementEvaluator->body();
	}
	
	#pragma oss taskwait
	
	
	/***********/
	/* PHASE 3 */
	/***********/
	
	tap.emitDiagnostic("*****************");
	tap.emitDiagnostic("***  PHASE 3  ***");
	tap.emitDiagnostic("***           ***");
	tap.emitDiagnostic("*** ", activeCPUs-1, " tests  ***");
	tap.emitDiagnostic("*****************");
	
	for (void *cpuIterator = nanos6_cpus_begin(); cpuIterator != nanos6_cpus_end(); cpuIterator = nanos6_cpus_advance(cpuIterator)) {
		long systemCPU = nanos6_cpus_get(cpuIterator);
		long virtualCPU = nanos6_cpus_get_virtual(cpuIterator);
		if (virtualCPU != currentVirtualCPU) {
			tap.evaluate(
				tasksPerVirtualCPU[virtualCPU] <= 1,
				"Check that disabled CPUs will at most run 1 task"
			);
			tap.emitDiagnostic("CPU ", systemCPU, " has run ", (int) tasksPerVirtualCPU[virtualCPU], " tasks after being disabled");
		}
	}
	
	timer.stop();
	
	tap.emitDiagnostic("Elapsed time: ", (long int) timer, " us");
	tap.end();
	
	return 0;
}
