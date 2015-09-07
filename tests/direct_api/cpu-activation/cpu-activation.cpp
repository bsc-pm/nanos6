#include "api/nanos6_rt_interface.h"

#include "tests/infrastructure/ProgramLifecycle.hpp"
#include "tests/infrastructure/TestAnyProtocolProducer.hpp"
#include "tests/infrastructure/Timer.hpp"

#include "executors/threads/CPUActivation.hpp"
#include "executors/threads/CPU.hpp"
#include "executors/threads/ThreadManagerDebuggingInterface.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "lowlevel/ConditionVariable.hpp"
#include "tasks/TaskDebuggingInterface.hpp"
#include "tasks/Task.hpp"

#include <cassert>


#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define __FILE_LINE__ (__FILE__ ":" TOSTRING(__LINE__))


#define VALIDATION_STEPS_PER_CPU 16


extern TestAnyProtocolProducer tap;


void shutdownTests()
{
}


class Blocker {
public:
	ConditionVariable _condVar;
	
	Blocker()
	{
	}
	
	void body()
	{
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
	CPU *_expectedCPU;
	std::vector<std::atomic<int>> &_tasksPerCPU;
	
	PlacementEvaluator(CPU *expectedCPU, std::vector<std::atomic<int>> &tasksPerCPU)
		: _expectedCPU(expectedCPU), _tasksPerCPU(tasksPerCPU)
	{
	}
	
	void body()
	{
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);
		
		CPU *cpu = currentThread->getHardwarePlace();
		assert(cpu != nullptr);
		
		// Weak check since we cannot guarantee that a CPU will not run (only) one task
		tap.evaluateWeak(
			cpu == _expectedCPU,
			"Check that when only one CPU is enabled, all tasks run in that CPU",
			"Cannot guarantee that one task will not get past the status transition"
		);
		if (cpu != _expectedCPU) {
			tap.emitDiagnostic("Expected ", _expectedCPU->_systemCPUId, " got ", cpu->_systemCPUId);
			tap.emitDiagnostic("CPU ", _expectedCPU->_systemCPUId, " in activation status ", (CPU::activation_status_t) _expectedCPU->_activationStatus);
			tap.emitDiagnostic("CPU ", cpu->_systemCPUId, " in activation status ", (CPU::activation_status_t) _expectedCPU->_activationStatus);
		}
		
		_tasksPerCPU[cpu->_systemCPUId]++;
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
	
	ThreadManagerDebuggingInterface::cpu_list_t &cpuList = ThreadManagerDebuggingInterface::getCPUListRef();
	
	int activeCPUs = 0;
	cpu_set_t const &cpuMask = ThreadManagerDebuggingInterface::getProcessCPUMaskRef();
	for (size_t systemCPUId = 0; systemCPUId < CPU_SETSIZE; systemCPUId++) {
		if (CPU_ISSET(systemCPUId, &cpuMask)) {
			activeCPUs++;
		}
	}
	
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
		/* Phase 0 */ 1
		/* Phase 1 */ + 6
		/* Phase 2 */ + activeCPUs*VALIDATION_STEPS_PER_CPU
		/* Phase 3 */ + (activeCPUs - 1)
	);
	tap.begin();
	
	
	/***********/
	/* PHASE 0 */
	/***********/
	
	// First ensure that the runtime has been fully initialized (that all CPUs have been enabled)
	tap.timedEvaluate(
		[&]() {
			int initializedCPUs = 0;
			for (CPU *cpu : cpuList) {
				if (cpu != nullptr) {
					initializedCPUs++;
				}
			}
			
			return (initializedCPUs == activeCPUs);
		},
		5000000, // 5 seconds
		"Check that the runtime initializes all the CPUs in a reasonable amount of time"
	); // 1
	
	
	Timer timer;
	
	
	/***********/
	/* PHASE 1 */
	/***********/
	
	Blocker *blocker = new Blocker();
	Blocker **blockerParam;
	void *blockerTask = nullptr;
	static nanos_task_invocation_info blocker_invocation_info = {
		__FILE_LINE__
	};
	nanos_create_task(&blocker_info, &blocker_invocation_info, sizeof(Blocker *), (void **) &blockerParam, &blockerTask);
	*blockerParam = blocker;
	nanos_submit_task(blockerTask);
	
	Task *actualRuntimeTask = TaskDebuggingInterface::getRuntimeTask(blockerTask);
	
	while (actualRuntimeTask->getThread() == nullptr) {
		// Wait
	}
	
	CPU *blockerCPU = actualRuntimeTask->getThread()->getHardwarePlace();
	assert(blockerCPU != nullptr);
	
	tap.evaluate(
		blockerCPU->_activationStatus == CPU::enabled_status,
		"Check that the CPU that runs a task is enabled"
	); // 2
	
	CPUActivation::disable(blockerCPU->_systemCPUId);
	tap.evaluate(
		blockerCPU->_activationStatus == CPU::disabling_status,
		"Check that attempting to disable a CPU will set it to disabling status"
	); // 3
	tap.bailOutAndExitIfAnyFailed();
	
	blocker->_condVar.signal();
	tap.timedEvaluate(
		[&]() {
			return (blockerCPU->_activationStatus == CPU::disabled_status);
		},
		1000000, // 1 second
		"Check that the CPU completes the deactivation in a reasonable amount of time"
	); // 4
	tap.bailOutAndExitIfAnyFailed();
	
	CPUActivation::disable(blockerCPU->_systemCPUId);
	tap.evaluate(
		blockerCPU->_activationStatus == CPU::disabled_status,
		"Check that attempting to disable an already disabled CPU keeps it untouched"
	); // 5
	
	CPUActivation::enable(blockerCPU->_systemCPUId);
	tap.timedEvaluate(
		[&]() {
			return (blockerCPU->_activationStatus == CPU::enabled_status);
		},
		1000000, // 1 second
		"Check that enabling a CPU will eventually set it to enabled"
	); // 6
	tap.bailOutAndExitIfAnyFailed();
	
	CPUActivation::enable(blockerCPU->_systemCPUId);
	tap.evaluate(
		blockerCPU->_activationStatus == CPU::enabled_status,
		"Check that reenabling a CPU does not change its status"
	); // 7
	tap.bailOutAndExitIfAnyFailed();
	
	nanos_taskwait(__FILE_LINE__);
	
	
	/***********/
	/* PHASE 2 */
	/***********/
	
	WorkerThread *thisThread = WorkerThread::getCurrentWorkerThread();
	assert(thisThread != nullptr);
	
	CPU *thisCPU = thisThread->getHardwarePlace();
	assert(thisCPU != nullptr);
	
	tap.emitDiagnostic("Will be using CPU ", thisCPU->_systemCPUId);
	
	// Disable all other CPUs
	for (CPU *cpu : cpuList) {
		if (cpu != nullptr) {
			if (cpu != thisCPU) {
				tap.emitDiagnostic("Disabling CPU ", cpu->_systemCPUId);
				CPUActivation::disable(cpu->_systemCPUId);
			} else {
				tap.emitDiagnostic("Not disabling CPU ", cpu->_systemCPUId);
			}
		}
	}
	
	std::vector<std::atomic<int>> tasksPerCPU(cpuList.size());
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
	
	for (CPU *cpu: cpuList) {
		if ((cpu != nullptr) && (cpu != thisCPU)) {
			tap.evaluate(
				tasksPerCPU[cpu->_systemCPUId] <= 1,
				"Check that disabled CPUs will at most run 1 task"
			);
			tap.emitDiagnostic("CPU ", cpu->_systemCPUId, " has run ", (int) tasksPerCPU[cpu->_systemCPUId], " tasks after being disabled");
		}
	}
	
	timer.stop();
	
	tap.emitDiagnostic("Elapsed time: ", (long int) timer, " us");
	
	shutdownTimer.start();
	
	return 0;
}
