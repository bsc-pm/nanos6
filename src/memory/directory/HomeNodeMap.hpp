/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef HOME_NODE_MAP_HPP
#define HOME_NODE_MAP_HPP

#include <vector>

#include <IntrusiveLinearRegionMap.hpp>
#include <IntrusiveLinearRegionMapImplementation.hpp>

#include "HomeMapEntry.hpp"
#include "lowlevel/PaddedTicketSpinLock.hpp"

class DataAccessRegion;
class MemoryPlace;

class HomeNodeMap : public IntrusiveLinearRegionMap<HomeMapEntry, boost::intrusive::function_hook< HomeMapEntryLinkingArtifacts > > {
	typedef IntrusiveLinearRegionMap<
		HomeMapEntry,
		boost::intrusive::function_hook<HomeMapEntryLinkingArtifacts>
	> BaseType;
	
	//! Lock to protect accesses to the Map
	typedef PaddedTicketSpinLock<int, 128> spinlock_t;
	spinlock_t lock;
public:
	
	//! \brief An auxiliary type to return info to callers
	typedef std::vector<HomeMapEntry *> HomeNodesArray;
	
	HomeNodeMap() : BaseType()
	{
	}
	
	~HomeNodeMap()
	{
	}
	
	//! \brief Insert a region in the map
	void insert(DataAccessRegion const &region,
		MemoryPlace const *homeNode);
	
	//! \brief Find the home nodes of a region
	HomeNodesArray *find(DataAccessRegion const &region);
	
	//! \brief Remove a region from the map
	void erase(DataAccessRegion const &region);
};

#endif /* HOME_NODE_MAP_HPP */
