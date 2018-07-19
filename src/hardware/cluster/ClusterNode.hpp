/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_NODE_HPP
#define CLUSTER_NODE_HPP

#include "hardware/places/ComputePlace.hpp"

class ClusterNode : public ComputePlace {
private:
	//! This the index of the node related to the communication layer
	int _commIndex;
	
public:
	ClusterNode(int index, int commIndex)
		: ComputePlace(index, nanos6_device_t::nanos6_host_device),
		_commIndex(commIndex)
	{
	}
	
	~ClusterNode()
	{
	}
	
	inline int getCommIndex() const
	{
		return _commIndex;
	}
};

#endif /* CLUSTER_NODE_HPP */
