#include "InstrumentComputePlaceManagement.hpp"
#include "executors/threads/CPUManager.hpp"

Instrument::compute_place_id_t Instrument::createdCPU(unsigned int virtualCPUId, __attribute__((unused)) size_t NUMANode)
{
	std::vector<CPU *> const &cpus = CPUManager::getCPUListReference();
	CPU *cpu = cpus[virtualCPUId];
	uint16_t systemCPUId = cpu->getSystemCPUId();
	return compute_place_id_t(systemCPUId);
}
