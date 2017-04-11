#ifndef CPU_MANAGER_HPP
#define CPU_MANAGER_HPP


#include <mutex>
#include <atomic>

#include <boost/dynamic_bitset.hpp>

#include "hardware/places/ComputePlace.hpp"
#include "lowlevel/SpinLock.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#include "CPU.hpp"


class CPUManager {
private:
	//! \brief CPU mask of the process
	static cpu_set_t _processCPUMask;
	
	//! \brief per-CPU data indexed by system CPU identifier
	static std::vector<CPU *> _cpus;
	
	//! \brief numer of initialized CPUs
	static size_t _totalCPUs;
	
	//! \brief indicates if the thread manager has finished initializing the CPUs
	static std::atomic<bool> _finishedCPUInitialization;
	
	//! \brief threads blocked due to idleness
	static boost::dynamic_bitset<> _idleCPUs;
	
	static SpinLock _idleCPUsLock;
	
	
public:
	static void preinitialize();
	
	static void initialize();
	
	//! \brief get the CPU object assigned to a given numerical system CPU identifier
	static inline CPU *getCPU(size_t systemCPUId);
	
	//! \brief get the maximum number of CPUs that will be used
	static inline long getTotalCPUs();
	
	//! \brief check if initialization has finished
	static inline bool hasFinishedInitialization();
	
	//! \brief get a reference to the CPU mask of the process
	static inline cpu_set_t const &getProcessCPUMaskReference();
	
	//! \brief get a reference to the list of CPUs
	static inline std::vector<CPU *> const &getCPUListReference();

	//! \brief mark a CPU as idle
	static inline void cpuBecomesIdle(CPU *cpu);

	//! \brief get an idle CPU
	static inline CPU *getIdleCPU();
};


inline CPU *CPUManager::getCPU(size_t systemCPUId)
{
	// TODO: make it simpler
	for (size_t i = 0; i < _cpus.size(); ++i) {
		if (_cpus[i]->_systemCPUId == systemCPUId) {
			return _cpus[i];
		}
	}

	assert(false);
	return nullptr;
}

inline long CPUManager::getTotalCPUs()
{
	return _totalCPUs;
}

inline bool CPUManager::hasFinishedInitialization()
{
	return _finishedCPUInitialization;
}


inline cpu_set_t const &CPUManager::getProcessCPUMaskReference()
{
	return _processCPUMask;
}

inline std::vector<CPU *> const &CPUManager::getCPUListReference()
{
	return _cpus;
}


inline void CPUManager::cpuBecomesIdle(CPU *cpu)
{
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	_idleCPUs[cpu->_virtualCPUId] = true;
}


inline CPU *CPUManager::getIdleCPU()
{
	std::lock_guard<SpinLock> guard(_idleCPUsLock);
	boost::dynamic_bitset<>::size_type idleCPU = _idleCPUs.find_first();
	if (idleCPU != boost::dynamic_bitset<>::npos) {
		_idleCPUs[idleCPU] = false;
		return _cpus[idleCPU];
	} else {
		return nullptr;
	}
}

#endif // THREAD_MANAGER_HPP
