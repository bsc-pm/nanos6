/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef __CLUSTER_MANAGER_HPP__
#define __CLUSTER_MANAGER_HPP__

#include <vector>
#include <string>

class ClusterNode;
class Messenger;

class ClusterManager {
private:
	//! Number of cluster nodes
	static int _clusterSize;
	
	/** A vector of all ClusterNodes in the system.
	 *
	 * We might need to make this a map later on, when we start
	 * adding/removing nodes
	 */
	static std::vector<ClusterNode *> _clusterNodes;
	
	//! ClusterNode object of the current node
	static ClusterNode *_thisNode;
	
	//! ClusterNode of the master node
	static ClusterNode *_masterNode;
	
	//! Messenger object for cluster communication.
	static Messenger *_msn;
	
	static void initializeCluster(std::string const &commType);
	
	//! private constructor. This is a singleton.
	ClusterManager()
	{}
public:
	static void initialize();
	static void shutdown();
	
	static inline ClusterNode *getClusterNode(int nodeId)
	{
		return _clusterNodes[nodeId];
	}
	
	static inline ClusterNode *getClusterNode()
	{
		return _thisNode;
	}
	
	static inline bool isMasterNode()
	{
		return _masterNode == _thisNode;
	}
	
	static inline int clusterSize()
	{
		return _clusterSize;
	}
	
	static inline bool inClusterMode()
	{
		return _clusterSize > 1;
	}
};

#endif /* __CLUSTER_MANAGER_HPP__ */
