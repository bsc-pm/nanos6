/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_LOCALITY_SCHEDULER_HPP
#define CLUSTER_LOCALITY_SCHEDULER_HPP

#include "scheduling/SchedulerInterface.hpp"
#include "system/RuntimeInfo.hpp"

#include <ClusterManager.hpp>

class ClusterLocalityScheduler : public SchedulerInterface {
	//! Current cluster node
	ClusterNode *_thisNode;
	
	//! Number of cluster nodes
	int _clusterSize;
	
public:
	ClusterLocalityScheduler()
	{
		RuntimeInfo::addEntry("cluster-scheduler", "Cluster Scheduler", getName());
		_thisNode = ClusterManager::getCurrentClusterNode();
		_clusterSize = ClusterManager::clusterSize();
	}
	
	~ClusterLocalityScheduler()
	{
	}
	
	void addReadyTask(Task *task, ComputePlace *computePlace,
			ReadyTaskHint hint = NO_HINT);
	
	inline std::string getName() const
	{
		return "ClusterLocalityScheduler";
	}
};

#endif // CLUSTER_LOCALITY_SCHEDULER_HPP
