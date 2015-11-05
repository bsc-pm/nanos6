#include "api/nanos6_rt_interface.h"
#include "api/nanos6_debug_interface.h"

#include "tests/infrastructure/ProgramLifecycle.hpp"
#include "tests/infrastructure/TestAnyProtocolProducer.hpp"
#include "tests/infrastructure/Timer.hpp"

#include "lowlevel/ConditionVariable.hpp"

#include <atomic>
#include <cassert>
#include <vector>

#include <sched.h>


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define __FILE_LINE__ (__FILE__ ":" TOSTRING(__LINE__))


#define VALIDATION_STEPS_PER_CPU 16


extern TestAnyProtocolProducer tap;


void shutdownTests()
{
}


static std::atomic<long> _blockerCPU;

class Blocker {
public:
	ConditionVariable _condVar;
	
	Blocker()
	{
	}
	
	void body()
	{
		_blockerCPU = nanos_get_current_system_cpu();
		_condVar.wait();
	}
};


static void blocker_wrapper(void *argsBlock)
{
	Blocker **blocker = (Blocker **) argsBlock;
	
	(*blocker)->body();
}

static void blocker_register_depinfo(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
{
}

static void blocker_register_copies(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
{
}

static nanos_task_info blocker_info = {
	blocker_wrapper,
	blocker_register_depinfo,
	blocker_register_copies,
	"blocker",
	"blocker_source_line"
};


class PlacementEvaluator {
public:
	long _expectedCPU;
	std::vector<std::atomic<int>> &_tasksPerCPU;
	
	PlacementEvaluator(long expectedCPU, std::vector<std::atomic<int>> &tasksPerCPU)
		: _expectedCPU(expectedCPU), _tasksPerCPU(tasksPerCPU)
	{
	}
	
	void body()
	{
		long cpu = nanos_get_current_system_cpu();
		
		// Weak check since we cannot guarantee that a CPU will not run (only) one task
		tap.evaluateWeak(
			cpu == _expectedCPU,
			"Check that when only one CPU is enabled, all tasks run in that CPU",
			"Cannot guarantee that one task will not get past the status transition"
		);
		if (cpu != _expectedCPU) {
			tap.emitDiagnostic("Expected ", _expectedCPU, " got ", cpu);
			tap.emitDiagnostic("CPU ", _expectedCPU, " in activation status ", nanos_get_cpu_status(_expectedCPU));
			tap.emitDiagnostic("CPU ", cpu, " in activation status ", nanos_get_cpu_status(_expectedCPU));
		}
		
		_tasksPerCPU[cpu]++;
	}
	
};


static void placement_evaluator_wrapper(void *argsBlock)
{
	PlacementEvaluator **placementEvaluator = (PlacementEvaluator **) argsBlock;
	
	(*placementEvaluator)->body();
}

static void placement_evaluator_register_depinfo(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
{
}

static void placement_evaluator_register_copies(__attribute__((unused)) void *handler, __attribute__((unused)) void *argsBlock)
{
}

static nanos_task_info placement_evaluator_info = {
	placement_evaluator_wrapper,
	placement_evaluator_register_depinfo,
	placement_evaluator_register_copies,
	"placement_evaluator",
	"placement_evaluator_source_line"
};


#include <sched.h>

int main(int argc, char **argv) {
	initializationTimer.stop();
	
	nanos_wait_for_full_initialization();
	
	long activeCPUs = nanos_get_num_cpus();
	if (activeCPUs == 1) {
		// This test only works correctly with more than 1 CPU
		tap.registerNewTests(1);
		tap.begin();
		tap.evaluateWeak(
			activeCPUs > 1,
			"Check that the test is being executed with more than 1 CPU",
			"This test does not work with just 1 CPU"
		);
		shutdownTimer.start();
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
	
	_blockerCPU = -1;
	Blocker *blocker = new Blocker();
	Blocker **blockerParam;
	void *blockerTask = nullptr;
	static nanos_task_invocation_info blocker_invocation_info = {
		__FILE_LINE__
	};
	nanos_create_task(&blocker_info, &blocker_invocation_info, sizeof(Blocker *), (void **) &blockerParam, &blockerTask);
	*blockerParam = blocker;
	nanos_submit_task(blockerTask);
	
	// Wait to get the blocker CPU
	while (_blockerCPU == -1) {
		sched_yield();
	}
	
	long blockerCPU = _blockerCPU;
	
	tap.evaluate(
		nanos_get_cpu_status(blockerCPU) == nanos_enabled_cpu,
		"Check that the CPU that runs a task is enabled"
	); // 2
	
	nanos_disable_cpu(blockerCPU);
	tap.evaluate(
		nanos_get_cpu_status(blockerCPU) == nanos_disabling_cpu,
		"Check that attempting to disable a CPU will set it to disabling status"
	); // 3
	tap.bailOutAndExitIfAnyFailed();
	
	blocker->_condVar.signal();
	tap.timedEvaluate(
		[&]() {
			return (nanos_get_cpu_status(blockerCPU) == nanos_disabled_cpu);
		},
		1000000, // 1 second
		"Check that the CPU completes the deactivation in a reasonable amount of time"
	); // 4
	tap.bailOutAndExitIfAnyFailed();
	
	nanos_disable_cpu(blockerCPU);
	tap.evaluate(
		nanos_get_cpu_status(blockerCPU) == nanos_disabled_cpu,
		"Check that attempting to disable an already disabled CPU keeps it untouched"
	); // 5
	
	nanos_enable_cpu(blockerCPU);
	tap.timedEvaluate(
		[&]() {
			return (nanos_get_cpu_status(blockerCPU) == nanos_enabled_cpu);
		},
		1000000, // 1 second
		"Check that enabling a CPU will eventually set it to enabled"
	); // 6
	tap.bailOutAndExitIfAnyFailed();
	
	nanos_enable_cpu(blockerCPU);
	tap.evaluate(
		nanos_get_cpu_status(blockerCPU) == nanos_enabled_cpu,
		"Check that reenabling a CPU does not change its status"
	); // 7
	tap.bailOutAndExitIfAnyFailed();
	
	nanos_taskwait(__FILE_LINE__);
	
	
	/***********/
	/* PHASE 2 */
	/***********/
	
	long thisCPU = nanos_get_current_system_cpu();
	
	tap.emitDiagnostic("Will be using CPU ", thisCPU);
	
	// Disable all other CPUs
	for (void *cpuIterator = nanos_cpus_begin(); cpuIterator != nanos_cpus_end(); cpuIterator = nanos_cpus_advance(cpuIterator)) {
		long cpu = nanos_cpus_get(cpuIterator);
		if (cpu != thisCPU) {
			tap.emitDiagnostic("Disabling CPU ", cpu);
			nanos_disable_cpu(cpu);
		} else {
			tap.emitDiagnostic("Not disabling CPU ", cpu);
		}
	}
	
	std::vector<std::atomic<int>> tasksPerCPU(activeCPUs);
	for (std::atomic<int> &tasks : tasksPerCPU) {
		tasks = 0;
	}
	
	for (int i=0; i < activeCPUs*VALIDATION_STEPS_PER_CPU; i++) {
		PlacementEvaluator *placementEvaluator = new PlacementEvaluator(thisCPU, tasksPerCPU);
		PlacementEvaluator **placementEvaluatorParam = nullptr;
		void *placementEvaluatorTask = nullptr;
		static nanos_task_invocation_info placement_evaluator_invocation_info = {
			__FILE_LINE__
		};
		nanos_create_task(&placement_evaluator_info, &placement_evaluator_invocation_info, sizeof(PlacementEvaluator *), (void **) &placementEvaluatorParam, &placementEvaluatorTask);
		*placementEvaluatorParam = placementEvaluator;
		nanos_submit_task(placementEvaluatorTask);
	}
	
	nanos_taskwait(__FILE_LINE__);
	
	
	/***********/
	/* PHASE 3 */
	/***********/
	
	for (void *cpuIterator = nanos_cpus_begin(); cpuIterator != nanos_cpus_end(); cpuIterator = nanos_cpus_advance(cpuIterator)) {
		long cpu = nanos_cpus_get(cpuIterator);
		if (cpu != thisCPU) {
			tap.evaluate(
				tasksPerCPU[cpu] <= 1,
				"Check that disabled CPUs will at most run 1 task"
			);
			tap.emitDiagnostic("CPU ", cpu, " has run ", (int) tasksPerCPU[cpu], " tasks after being disabled");
		}
	}
	
	timer.stop();
	
	tap.emitDiagnostic("Elapsed time: ", (long int) timer, " us");
	
	shutdownTimer.start();
	
	return 0;
}
