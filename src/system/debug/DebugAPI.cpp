#include <cassert>

#include <nanos6/debug.h>
#include "executors/threads/CPUActivation.hpp"
#include "executors/threads/ThreadManager.hpp"
#include "executors/threads/WorkerThread.hpp"
#include "tasks/Task.hpp"


void nanos_wait_for_full_initialization(void)
{
	while (!CPUManager::hasFinishedInitialization()) {
		// Wait
	}
}

unsigned int nanos_get_num_cpus(void)
{
	return CPUManager::getTotalCPUs();
}

long nanos_get_current_system_cpu()
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	
	assert(currentThread != 0);
	CPU *currentCPU = currentThread->getComputePlace();
	
	assert(currentCPU != 0);
	return currentCPU->_systemCPUId;
}

unsigned int nanos_get_current_virtual_cpu()
{
	WorkerThread *currentThread = WorkerThread::getCurrentWorkerThread();
	
	if (currentThread == nullptr) {
		return 0;
	}
	
	CPU *currentCPU = currentThread->getComputePlace();
	assert(currentCPU != 0);
	
	return currentCPU->_virtualCPUId;
}

void nanos_enable_cpu(long systemCPUId)
{
	CPUActivation::enable(systemCPUId);
}

void nanos_disable_cpu(long systemCPUId)
{
	CPUActivation::disable(systemCPUId);
}


nanos_cpu_status_t nanos_get_cpu_status(long systemCPUId)
{
	CPU *cpu = CPUManager::getCPU(systemCPUId);
	
	assert(cpu != 0);
	switch (cpu->_activationStatus.load()) {
		//TODO: FIXME: IS THIS CORRECT? Just introduced to fix a compilation warning.
		case CPU::uninitialized_status:
			return nanos_disabled_cpu; 
		case CPU::starting_status:
			return nanos_starting_cpu;
		case CPU::enabling_status:
			return nanos_enabling_cpu;
		case CPU::enabled_status:
			return nanos_enabled_cpu;
		case CPU::disabling_status:
			return nanos_disabling_cpu;
		case CPU::disabled_status:
			return nanos_disabled_cpu;
	}
	
	assert("Unknown CPU status" == 0);
	return nanos_invalid_cpu_status;
}


#if 0
void nanos_wait_until_task_starts(void *taskHandle)
{
	assert(taskHandle != 0);
	
	Task *task = (Task *) taskHandle;
	while (task->getThread() == 0) {
		// Wait
	}
}


long nanos_get_system_cpu_of_task(void *taskHandle)
{
	assert(taskHandle != 0);
	
	Task *task = (Task *) taskHandle;
	WorkerThread *thread = task->getThread();
	
	assert(thread != 0);
	CPU *cpu = thread->getComputePlace();
	
	assert(cpu != 0);
	return cpu->_systemCPUId;
}
#endif


typedef std::vector<CPU *>::const_iterator cpu_iterator_t;


static void *nanos_cpus_skip_uninitialized(void *cpuIterator) {
	std::vector<CPU *> const &cpuList = CPUManager::getCPUListReference();
	
	cpu_iterator_t *itp = (cpu_iterator_t *) cpuIterator;
	if (itp == 0) {
		return 0;
	}
	
	do {
		if ((*itp) == cpuList.end()) {
			delete itp;
			return 0;
		}
		
		CPU *cpu = *(*itp);
		
		if ((cpu != 0) && (cpu->_activationStatus != CPU::uninitialized_status)) {
			return itp;
		}
		
		(*itp)++;
	} while (true);
}


void *nanos_cpus_begin(void)
{
	std::vector<CPU *> const &cpuList = CPUManager::getCPUListReference();
	cpu_iterator_t it = cpuList.begin();
	
	if (it == cpuList.end()) {
		return 0;
	} else {
		void *result = new cpu_iterator_t(it);
		return nanos_cpus_skip_uninitialized(result);
	}
}

void *nanos_cpus_end(void)
{
	return 0;
}

void *nanos_cpus_advance(void *cpuIterator)
{
	cpu_iterator_t *itp = (cpu_iterator_t *) cpuIterator;
	if (itp == 0) {
		return 0;
	}
	
	(*itp)++;
	return nanos_cpus_skip_uninitialized(itp);
}

long nanos_cpus_get(void *cpuIterator)
{
	cpu_iterator_t *it = (cpu_iterator_t *) cpuIterator;
	assert (it != 0);
	
	CPU *cpu = *(*it);
	assert(cpu != 0);
	
	return cpu->_systemCPUId;
}



