/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CLUSTER_MEMORY_MANAGEMENT_HPP
#define CLUSTER_MEMORY_MANAGEMENT_HPP

#include <nanos6/cluster.h>

namespace ClusterMemoryManagement {
	void *dmalloc(
		size_t size,
		nanos6_data_distribution_t policy,
		size_t numDimensions,
		size_t *dimensions
	);
	
	void dfree(void *ptr, size_t size);
	
	void *lmalloc(size_t size);
	
	void lfree(void *ptr, size_t size);
}

#endif /* CLUSTER_MEMORY_MANAGEMENT_HPP */
