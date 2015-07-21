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


#define CPU_ACTIVATION_VALIDATION_STEPS 50


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
	
	Task2(CPU *expectedCPU)
		: Task(nullptr), _expectedCPU(expectedCPU)
	{
	}
	
	virtual void body()
	{
		WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
		assert(currentThread != nullptr);
		
		CPU *cpu = currentThread->getHardwarePlace();
		assert(cpu != nullptr);
		
		tap.evaluate(cpu == _expectedCPU, "Check that when only one CPU is enabled, all tasks run in that CPU");
	}
	
};


#include <sched.h>

int main(int argc, char **argv) {
	initializationTimer.stop();
	
	tap.registerNewTests(6 + CPU_ACTIVATION_VALIDATION_STEPS);
	tap.begin();
	
	Timer timer;
	
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
	); // 1
	
	CPUActivation::disable(task1CPU->_systemCPUId);
	tap.evaluate(
		task1CPU->_activationStatus == CPU::disabling_status,
		"Check that attempting to disable a CPU will set it to disabling status"
	); // 2
	tap.bailOutAndExitIfAnyFailed();
	
	task1->_condVar.signal();
	tap.timedEvaluate(
		[&]() {
			return (task1CPU->_activationStatus == CPU::disabled_status);
		},
		1000000, // 1 second
		"Check that the CPU completes the deactivation in a reasonable amount of time"
	); // 3
	tap.bailOutAndExitIfAnyFailed();
	
	CPUActivation::disable(task1CPU->_systemCPUId);
	tap.evaluate(
		task1CPU->_activationStatus == CPU::disabled_status,
		"Check that attempting to disable an already disabled CPU keeps it untouched"
	); // 4
	
	CPUActivation::enable(task1CPU->_systemCPUId);
	tap.timedEvaluate(
		[&]() {
			return (task1CPU->_activationStatus == CPU::enabled_status);
		},
		1000000, // 1 second
		"Check that enabling a CPU will eventually set it to enabled"
	); // 5
	tap.bailOutAndExitIfAnyFailed();
	
	CPUActivation::enable(task1CPU->_systemCPUId);
	tap.evaluate(
		task1CPU->_activationStatus == CPU::enabled_status,
		"Check that reenabling a CPU does not change its status"
	); // 6
	tap.bailOutAndExitIfAnyFailed();
	
	ompss::taskWait();
	
	WorkerThread *thisThread = WorkerThread::getCurrentWorkerThread();
	assert(thisThread != nullptr);
	
	CPU *thisCPU = thisThread->getHardwarePlace();
	assert(thisCPU != nullptr);
	
	// Disable all other CPUs
	for (CPU *cpu : ThreadManagerDebuggingInterface::getCPUListRef()) {
		if (cpu != nullptr) {
			if (cpu != thisCPU) {
				CPUActivation::disable(cpu->_systemCPUId);
			}
		}
	}
	
	for (int i=0; i < CPU_ACTIVATION_VALIDATION_STEPS; i++) {
		Task2 *task2Instance = new Task2(thisCPU);
		ompss::addTask(task2Instance);
	}
	
	ompss::taskWait();
	
	timer.stop();
	
	tap.emitDiagnostic("Elapsed time: ", (long int) timer, " us");
	
	shutdownTimer.start();
	
	return 0;
}
