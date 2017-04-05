#include <boost/dynamic_bitset.hpp>

#include "CPU.hpp"
#include "CPUManager.hpp"
#include "ThreadManager.hpp"

cpu_set_t CPUManager::_processCPUMask;
std::vector<CPU *> CPUManager::_cpus;
size_t CPUManager::_totalCPUs;
std::atomic<bool> CPUManager::_finishedCPUInitialization;
SpinLock CPUManager::_idleCPUsLock;
boost::dynamic_bitset<> CPUManager::_idleCPUs;

void CPUManager::preinitialize()
{
	_finishedCPUInitialization = false;
	_totalCPUs = 0;
	
	int rc = sched_getaffinity(0, sizeof(cpu_set_t), &_processCPUMask);
	FatalErrorHandler::handle(rc, " when retrieving the affinity of the current pthread ", pthread_self());
	
	std::vector<ComputePlace *> cpus = HardwareInfo::getComputeNodes();

	_cpus.resize(cpus.size());
	for (size_t i = 0; i < cpus.size(); ++i) {
		_cpus[i] = (CPU *)cpus[i];
	}

	_idleCPUs.resize(cpus.size());
	_idleCPUs.reset();
}

void CPUManager::initialize()
{
	// Start a thread in each CPU enabled
	for (size_t systemCPUId = 0; systemCPUId < CPU_SETSIZE; systemCPUId++) {
		if (CPU_ISSET(systemCPUId, &_processCPUMask)) {
			assert(_cpus[systemCPUId] != nullptr);
			
			_cpus[systemCPUId]->initializeIfNeeded();
			ThreadManager::initializeThread(_cpus[systemCPUId]);
			_totalCPUs++;
		}
	}
	
	_finishedCPUInitialization = true;
}
