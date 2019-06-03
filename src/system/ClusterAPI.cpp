/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/cluster.h>

#include <ClusterManager.hpp>
#include <ClusterMemoryManagement.hpp>
#include <ClusterNode.hpp>


extern "C" int nanos6_in_cluster_mode(void)
{
	return ClusterManager::inClusterMode();
}

extern "C" int nanos6_is_master_node(void)
{
	return ClusterManager::isMasterNode();
}

extern "C" int nanos6_get_cluster_node_id(void)
{
	return ClusterManager::getCurrentClusterNode()->getIndex();
}

extern "C" int nanos6_get_num_cluster_nodes(void)
{
	return ClusterManager::clusterSize();
}

extern "C" void *nanos6_dmalloc(
	size_t size,
	nanos6_data_distribution_t policy,
	size_t num_dimensions,
	size_t *dimensions
) {
	if (size == 0) {
		return nullptr;
	}
	
	return ClusterMemoryManagement::dmalloc(size, policy, num_dimensions, dimensions);
}

extern "C" void nanos6_dfree(void *ptr, size_t size)
{
	ClusterMemoryManagement::dfree(ptr, size);
}

extern "C" void *nanos6_lmalloc(size_t size)
{
	if (size == 0) {
		return nullptr;
	}
	
	return ClusterMemoryManagement::lmalloc(size);
}

extern "C" void nanos6_lfree(void *ptr, size_t size)
{
	ClusterMemoryManagement::lfree(ptr, size);
}
