/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_MEMORY_NODE_HPP
#define CLUSTER_MEMORY_NODE_HPP

#include "hardware/places/MemoryPlace.hpp"

class ClusterMemoryNode : public MemoryPlace {
	//! This is the index of the node related to the communication layer
	int _commIndex;
	
public:
	ClusterMemoryNode(int index, int commIndex)
		: MemoryPlace(index, nanos6_device_t::nanos6_cluster_device),
		_commIndex(commIndex)
	{
	}
	
	~ClusterMemoryNode()
	{
	}
	
	//! \brief Get the communicator index of the ClusterMemoryNode
	inline int getCommIndex() const
	{
		return _commIndex;
	}
};


#endif /* CLUSTER_MEMORY_NODE_HPP */
