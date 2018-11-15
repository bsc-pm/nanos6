/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/cluster.h>

#include <ClusterManager.hpp>
#include <ClusterNode.hpp>


extern "C" int nanos6_in_cluster_mode()
{
	return ClusterManager::inClusterMode();
}

extern "C" int nanos6_is_master_node()
{
	return ClusterManager::isMasterNode();
}

extern "C" int nanos6_get_cluster_node_id()
{
	return ClusterManager::getCurrentClusterNode()->getIndex();
}

extern "C" int nanos6_get_num_cluster_nodes()
{
	return ClusterManager::clusterSize();
}
