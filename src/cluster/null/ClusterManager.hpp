/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef __CLUSTER_MANAGER_HPP__
#define __CLUSTER_MANAGER_HPP__

#include <vector>
#include <string>

#include <ClusterNode.hpp>

class ClusterManager {
	//! private constructor. This is a singleton.
	ClusterManager()
	{}
public:
	static inline void initialize()
	{
	}
	
	static inline void shutdown()
	{
	}
	
	static inline ClusterNode *getClusterNode(__attribute__((unused)) int nodeId = 0)
	{
		static ClusterNode ourDummyNode;
		return &ourDummyNode;
	}
	
	static inline bool isMasterNode()
	{
		return true;
	}
	
	static inline int clusterSize()
	{
		return 1;
	}
	
	static inline bool inClusterMode()
	{
		return false;
	}
};

#endif /* __CLUSTER_MANAGER_HPP__ */
