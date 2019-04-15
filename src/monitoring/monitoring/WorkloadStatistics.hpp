/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef WORKLOAD_STATISTICS_HPP
#define WORKLOAD_STATISTICS_HPP

#include <atomic>
#include <cstddef>


enum workload_t {
	instantiated_load = 0,
	blocked_load,
	ready_load,
	executing_load,
	finished_load,
	num_workloads,
	null_workload = -1
};

char const * const workloadDescriptions[num_workloads] = {
	"Instantiated Workload",
	"Blocked Workload",
	"Ready Workload",
	"Executing Workload",
	"Finished Workload"
};

class WorkloadStatistics {

private:
	
	//! Aggregated computational cost of a tasktype
	std::atomic<size_t> _accumulatedCost[num_workloads];
	
	
public:
	
	inline WorkloadStatistics()
	{
		for (unsigned short loadId = 0; loadId < num_workloads; ++loadId) {
			_accumulatedCost[loadId] = 0;
		}
	}
	
	
	//! \brief Increase the accumulated cost of a workload by a specific value
	//! \param loadId The workload's id
	//! \param cost The value
	inline void increaseAccumulatedCost(workload_t loadId, size_t cost)
	{
		_accumulatedCost[loadId] += cost;
	}
	
	//! \brief Decrease the accumulated cost of a workload by a specific value
	//! \param loadId The workload's id
	//! \param cost The value
	inline void decreaseAccumulatedCost(workload_t loadId, size_t cost)
	{
		_accumulatedCost[loadId] -= cost;
	}
	
	//! \brief Get the accumulated cost of a workload
	//! \param loadId The workload's id
	inline size_t getAccumulatedCost(workload_t loadId) const
	{
		return _accumulatedCost[loadId].load();
	}
	
};

#endif // WORKLOAD_STATISTICS_HPP
