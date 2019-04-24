/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_NODE_HPP
#define CLUSTER_NODE_HPP

#include "hardware/places/ComputePlace.hpp"

#include <ClusterMemoryNode.hpp>

class ClusterNode : public ComputePlace {
private:
	//! MemoryPlace associated with this cluster node
	ClusterMemoryNode *_memoryNode;
	
	//! This is the index of the node related to the
	//! communication layer
	int _commIndex;
	
public:
	ClusterNode(int index, int commIndex)
		: ComputePlace(index, nanos6_device_t::nanos6_cluster_device),
		_commIndex(commIndex)
	{
		_memoryNode = new ClusterMemoryNode(index, commIndex);
	}
	
	~ClusterNode()
	{
	}
	
	//! \brief Get the MemoryNode of the cluster node
	inline ClusterMemoryNode *getMemoryNode() const
	{
		return _memoryNode;
	}
	
	//! \brief Get the index of the ClusterNode
	inline int getIndex() const
	{
		return _index;
	}
	
	//! \brief Get the communicator index of the ClusterNode
	inline int getCommIndex() const
	{
		return _commIndex;
	}
};


#endif /* CLUSTER_NODE_HPP */
