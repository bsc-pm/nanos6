/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_MANAGER_HPP
#define CLUSTER_MANAGER_HPP

#include <vector>
#include <string>

#include <ClusterNode.hpp>

class ClusterManager {
	//! private constructor. This is a singleton.
	ClusterManager()
	{}
public:
	class ShutdownCallback
	{
	};
	
	static inline void initialize()
	{
	}
	
	static inline void shutdown()
	{
	}
	
	static inline ClusterNode *getClusterNode(__attribute__((unused)) int id)
	{
		static ClusterNode ourDummyNode;
		return &ourDummyNode;
	}
	
	static inline ClusterNode *getCurrentClusterNode()
	{
		static ClusterNode ourDummyNode;
		return &ourDummyNode;
	}
	
	static inline ClusterMemoryNode *getMemoryNode(__attribute__((unused)) int id)
	{
		static ClusterMemoryNode ourDummyNode;
		return &ourDummyNode;
	}
	
	static inline ClusterMemoryNode *getCurrentMemoryNode()
	{
		static ClusterMemoryNode ourDummyNode;
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
	
	static inline void setShutdownCallback(
		__attribute__((unused)) void (*func)(void *),
		__attribute__((unused)) void *args)
	{
	}
	
	static inline ShutdownCallback *getShutdownCallback()
	{
		return nullptr;
	}
};

#endif /* CLUSTER_MANAGER_HPP */
