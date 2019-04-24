/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_MEMORY_NODE_HPP
#define CLUSTER_MEMORY_NODE_HPP

#include "hardware/places/MemoryPlace.hpp"

class ClusterMemoryNode : public MemoryPlace {
public:
	ClusterMemoryNode(__attribute__((unused)) int index = 0, __attribute__((unused)) int commIndex = 0)
		: MemoryPlace(index, nanos6_device_t::nanos6_host_device)
	{
	}
	
	~ClusterMemoryNode()
	{
	}
	
	inline int getCommIndex() const
	{
		return 0;
	}
};

#endif /* CLUSTER_MEMORY_NODE_HPP */
