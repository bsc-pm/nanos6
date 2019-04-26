/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DISTRIBUTION_POLICY_HPP
#define DISTRIBUTION_POLICY_HPP

#include <vector>

#include <nanos6/cluster.h>

class DataAccessRegion;

namespace ClusterDirectory {
	//! \brief Register a DataAccessRegion in the Directory
	//!
	//! \param[in] region is the DataAccessRegion to allocate
	//! \param[in] policy is the policy to distribute memory across nodes
	//! \param[in] nrDimensions is the number of policy dimensions
	//! \param[in] dimensions is the dimensions of the distribution policy
	void registerAllocation(
		DataAccessRegion const &region,
		nanos6_data_distribution_t policy,
		size_t nrDimensions,
		size_t *dimensions
	);
	
	//! \brief Unregister a DataAccessRegion from the Directory
	//!
	//! \param[in] region is the region we unregister
	void unregisterAllocation(DataAccessRegion const &region);
}




#endif /* DISTRIBUTION_POLICY_HPP */
