/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEPENDENCY_SYSTEM_HPP
#define DEPENDENCY_SYSTEM_HPP

#include "CPUDependencyData.hpp"
#include "scheduling/SchedulerSupport.hpp"
#include "system/RuntimeInfo.hpp"

class DependencySystem {
public:
	static void initialize()
	{
		RuntimeInfo::addEntry("dependency_implementation", "Dependency Implementation", "discrete");

		size_t pow2CPUs = SchedulerSupport::roundToNextPowOf2(CPUManager::getTotalCPUs());
		TaskList::_actualChunkSize = std::min(TaskList::getMaxChunkSize(), pow2CPUs * 2);
		assert(SchedulerSupport::isPowOf2(TaskList::_actualChunkSize));
	}
};

#endif // DEPENDENCY_SYSTEM_HPP
