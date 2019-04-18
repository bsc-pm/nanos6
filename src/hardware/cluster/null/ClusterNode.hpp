/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_NODE_HPP
#define CLUSTER_NODE_HPP

#include "hardware/places/ComputePlace.hpp"

#include <ClusterMemoryNode.hpp>

class ClusterNode : public ComputePlace {
public:
	ClusterNode(__attribute__((unused))int index = 0, __attribute__((unused))int commIndex = 0)
		: ComputePlace(index, nanos6_device_t::nanos6_cluster_device)
	{
	}
	
	~ClusterNode()
	{
	}
	
	inline ClusterMemoryNode *getMemoryNode() const
	{
		static ClusterMemoryNode ourDummyNode;
		return &ourDummyNode;
	}
	
	inline int getIndex() const
	{
		return 0;
	}
	
	inline int getCommIndex() const
	{
		return 0;
	}
};


#endif /* CLUSTER_NODE_HPP */
