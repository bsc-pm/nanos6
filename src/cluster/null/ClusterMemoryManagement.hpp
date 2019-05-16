/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_MEMORY_MANAGEMENT_HPP
#define CLUSTER_MEMORY_MANAGEMENT_HPP

#include <cstdlib>

#include <nanos6/cluster.h>

namespace ClusterMemoryManagement {
	inline void *dmalloc(size_t size, nanos6_data_distribution_t, size_t, size_t *)
	{
		return malloc(size);
	}
	
	inline void dfree(void *ptr, size_t)
	{
		free(ptr);
	}
	
	inline void *lmalloc(size_t size)
	{
		return malloc(size);
	}
	
	inline void lfree(void *ptr, size_t)
	{
		free(ptr);
	}
}

#endif /* CLUSTER_MEMORY_MANAGEMENT_HPP */
