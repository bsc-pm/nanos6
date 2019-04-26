/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include <DataAccessRegion.hpp>

#include "HomeMapEntry.hpp"
#include "HomeNodeMap.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "hardware/places/MemoryPlace.hpp"

void HomeNodeMap::insert(
	DataAccessRegion const &region,
	MemoryPlace const *homeNode
) {
	std::lock_guard<spinlock_t> guard(lock);
	processIntersectingAndMissing(
		region,
		[&] (HomeNodeMap::iterator pos) -> bool {
			HomeMapEntry *entry = &(*pos);
			
			FatalErrorHandler::failIf(true,
				"We do not support updating the home node ",
				"of a region. (Region: ",
				entry->getAccessRegion(),
				")"
			);
			
			//! Just to avoid compilation warnings
			return true;
		},
		[&] (DataAccessRegion missingRegion) -> bool {
			HomeMapEntry *entry = 
				new HomeMapEntry(missingRegion, homeNode);
			BaseType::insert(*entry);
			
			return true;
		}
	);
}

HomeNodeMap::HomeNodesArray *
HomeNodeMap::find(DataAccessRegion const &region)
{
	HomeNodesArray *ret = new HomeNodesArray();
	
	std::lock_guard<spinlock_t> guard(lock);
	processIntersectingAndMissing(
		region,
		[&] (HomeNodeMap::iterator pos) -> bool {
			HomeMapEntry *entry = &(*pos);
			ret->push_back(entry);
			return true;
		},
		[&] (DataAccessRegion missingRegion) -> bool 
		{
			FatalErrorHandler::failIf(
				true,
				"Asking the home node of an unkown ",
				"region: ",
				missingRegion
			);
			
			//! Just to avoid compilation warnings
			return true;
		}
	);
	
	return ret;
}

void HomeNodeMap::erase(DataAccessRegion const &region)
{
	std::lock_guard<spinlock_t> guard(lock);
	processIntersectingAndMissing(
		region,
		[&] (HomeNodeMap::iterator pos) -> bool {
			HomeMapEntry *entry = &(*pos);
			BaseType::erase(entry);	
			return true;
		},
		[&] (DataAccessRegion missingRegion) -> bool 
		{
			FatalErrorHandler::failIf(
				true,
				"Trying to erase an unknown home node",
				" mapping for region: ",
				missingRegion
			);
			
			//! Just to avoid compilation warnings
			return true;
		}
	);
}
