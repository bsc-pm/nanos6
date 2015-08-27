#include "system/ompss/AddTask.hpp"
#include "system/ompss/TaskWait.hpp"

#include "tests/infrastructure/ProgramLifecycle.hpp"
#include "tests/infrastructure/TestAnyProtocolProducer.hpp"
#include "tests/infrastructure/Timer.hpp"

#include "executors/threads/CPU.hpp"
#include "executors/threads/ThreadManagerDebuggingInterface.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "lowlevel/ConditionVariable.hpp"
#include "executors/threads/CPUActivation.hpp"

#include <cassert>


#define VALIDATION_STEPS_PER_CPU 16


extern TestAnyProtocolProducer tap;


void shutdownTests()
{
}


class Task1: public Task {
public:
	ConditionVariable _condVar;
	
	Task1():
		Task(nullptr)
	{
	}
	
	virtual void body()
	{
		_condVar.wait();
	}
};


class Task2: public Task {
public:
	CPU *_expectedCPU;
	std::vector<std::atomic<int>> &_tasksPerCPU;
	
	Task2(CPU *expectedCPU, std::vector<std::atomic<int>> &tasksPerCPU)
		: Task(nullptr), _expectedCPU(expectedCPU), _tasksPerCPU(tasksPerCPU)
	{
	}
	
	virtual void body()
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
	
	Task1 *task1 = new Task1();
	ompss::addTask(task1);
	
	while (task1->getThread() == nullptr) {
		// Wait
	}
	
	CPU *task1CPU = task1->getThread()->getHardwarePlace();
	assert(task1CPU != nullptr);
	
	tap.evaluate(
		task1CPU->_activationStatus == CPU::enabled_status,
		"Check that the CPU that runs a task is enabled"
	); // 2
	
	CPUActivation::disable(task1CPU->_systemCPUId);
	tap.evaluate(
		task1CPU->_activationStatus == CPU::disabling_status,
		"Check that attempting to disable a CPU will set it to disabling status"
	); // 3
	tap.bailOutAndExitIfAnyFailed();
	
	task1->_condVar.signal();
	tap.timedEvaluate(
		[&]() {
			return (task1CPU->_activationStatus == CPU::disabled_status);
		},
		1000000, // 1 second
		"Check that the CPU completes the deactivation in a reasonable amount of time"
	); // 4
	tap.bailOutAndExitIfAnyFailed();
	
	CPUActivation::disable(task1CPU->_systemCPUId);
	tap.evaluate(
		task1CPU->_activationStatus == CPU::disabled_status,
		"Check that attempting to disable an already disabled CPU keeps it untouched"
	); // 5
	
	CPUActivation::enable(task1CPU->_systemCPUId);
	tap.timedEvaluate(
		[&]() {
			return (task1CPU->_activationStatus == CPU::enabled_status);
		},
		1000000, // 1 second
		"Check that enabling a CPU will eventually set it to enabled"
	); // 6
	tap.bailOutAndExitIfAnyFailed();
	
	CPUActivation::enable(task1CPU->_systemCPUId);
	tap.evaluate(
		task1CPU->_activationStatus == CPU::enabled_status,
		"Check that reenabling a CPU does not change its status"
	); // 7
	tap.bailOutAndExitIfAnyFailed();
	
	ompss::taskWait();
	
	
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
		Task2 *task2Instance = new Task2(thisCPU, tasksPerCPU);
		ompss::addTask(task2Instance);
	}
	
	ompss::taskWait();
	
	
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
