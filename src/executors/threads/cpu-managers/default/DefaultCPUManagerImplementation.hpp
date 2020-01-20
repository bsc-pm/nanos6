/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEFAULT_CPU_MANAGER_IMPLEMENTATION_HPP
#define DEFAULT_CPU_MANAGER_IMPLEMENTATION_HPP

#include "executors/threads/CPUManagerInterface.hpp"


class DefaultCPUManagerImplementation : public CPUManagerInterface {

//! NOTE: Documentation for methods available in CPUManagerInterface.hpp
public:

	void shutdownPhase1();

	inline void shutdownPhase2()
	{
		// No need to destroy any structures in the default implementation
	}


	/*    CPUACTIVATION BRIDGE    */

	CPU::activation_status_t checkCPUStatusTransitions(WorkerThread *thread);

	bool acceptsWork(CPU *cpu);

	bool enable(size_t systemCPUId);

	bool disable(size_t systemCPUId);

	inline size_t getNumCPUsPerTaskforGroup() const
	{
		return _cpus.size() / _taskforGroups;
	}

};

#endif // DEFAULT_CPU_MANAGER_IMPLEMENTATION_HPP
