/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_RANDOM_SCHEDULER_HPP
#define CLUSTER_RANDOM_SCHEDULER_HPP

#include "scheduling/SchedulerInterface.hpp"
#include "system/RuntimeInfo.hpp"

#include <ClusterManager.hpp>

class ClusterRandomScheduler : public SchedulerInterface {
	//! Current cluster node
	ClusterNode *_thisNode;
	
	//! Number of cluster nodes
	int _clusterSize;
	
public:
	ClusterRandomScheduler()
	{
		RuntimeInfo::addEntry("cluster-scheduler", "Cluster Scheduler", getName());
		_thisNode = ClusterManager::getCurrentClusterNode();
		_clusterSize = ClusterManager::clusterSize();
	}
	
	~ClusterRandomScheduler()
	{
	}
	
	void addReadyTask(Task *task, ComputePlace *computePlace,
			ReadyTaskHint hint = NO_HINT);
	
	inline std::string getName() const
	{
		return "ClusterRandomScheduler";
	}
};

#endif // CLUSTER_RANDOM_SCHEDULER_HPP
