/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "memory/directory/Directory.hpp"
#include "DistributionPolicy.hpp"
#include "hardware/places/MemoryPlace.hpp"
#include "lowlevel/FatalErrorHandler.hpp"

#include <ClusterManager.hpp>
#include <DataAccessRegion.hpp>


namespace ClusterDirectory {
	static void registerAllocationEqupart(DataAccessRegion const &region)
	{
		void *address = region.getStartAddress();
		size_t size = region.getSize();
		size_t clusterSize = ClusterManager::clusterSize();
		
		size_t blockSize = size / clusterSize;
		size_t residual = size % clusterSize;
		size_t numBlocks = (blockSize > 0) ? clusterSize : 0;
		
		char *ptr = (char *)address;
		for (size_t i = 0; i < numBlocks; ++i) {
			DataAccessRegion newRegion((void *)ptr, blockSize);
			ClusterMemoryNode *homeNode = ClusterManager::getMemoryNode(i);
			Directory::insert(newRegion, homeNode);
			ptr += blockSize;
		}
		
		//! Add an extra entry to the first node for any residual
		//! uncovered region.
		if (residual > 0) {
			DataAccessRegion newRegion((void *)ptr, residual);
			ClusterMemoryNode *homeNode = ClusterManager::getMemoryNode(0);
			assert(homeNode != nullptr);
			
			Directory::insert(newRegion, homeNode);
		}
	}
	
	void registerAllocation(DataAccessRegion const &region,
			nanos6_data_distribution_t policy,
			__attribute__((unused)) size_t nrDimensions,
			__attribute__((unused)) size_t *dimensions)
	{
		switch (policy) {
			case nanos6_equpart_distribution:
				assert(nrDimensions == 0);
				assert(dimensions == nullptr);
				
				registerAllocationEqupart(region);
				break;
			case nanos6_block_distribution:
			case nanos6_cyclic_distribution:
			default:
				FatalErrorHandler::failIf(
					true,
					"Unknown distribution policy"
				);
		}
	}
	
	void unregisterAllocation(DataAccessRegion const &region)
	{
		//! Erase from Directory
		Directory::erase(region);
	}
}
