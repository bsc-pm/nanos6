/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef DIRECTORY_HPP
#define DIRECTORY_HPP

#include <cstdint>
#include <vector>

#include "HomeNodeMap.hpp"
#include "hardware/places/MemoryPlace.hpp"

class DataAccessRegion;

class Directory {
	//! \brief A map of the home nodes of memory regions
	static HomeNodeMap _homeNodes;
	
	//! \brief A MemoryPlace that 'points' to the Directory
	//!
	//! We use this MemoryPlace in every case we mean that
	//! we need to consult with the Directory in order to ask
	//! the actual location of a DataAccess
	static MemoryPlace _directoryMemoryPlace;
	
public:
	//! Exposing a type that describes the array of home nodes for a given
	//! memory region
	typedef HomeNodeMap::HomeNodesArray HomeNodesArray;
	
	//! \brief Check if a MemoryPlace is the Directory MemoryPlace
	//!
	//! \param[in] memoryPlace is the MemoryPlace we are checking
	//!
	//! \returns true if memoryPlace is the Directory MemoryPlace
	static inline bool isDirectoryMemoryPlace(
			MemoryPlace const *memoryPlace)
	{
		return (memoryPlace == &_directoryMemoryPlace);
	}
	
	//! \brief Check if the id is the id of the Directory MemoryPlace
	//!
	//! \param[in] index is the MemoryPlace index of the MemoryPlace we are
	//!		checking
	//!
	//! \returns true if the index corresponds to the Directory MemoryPlace
	static inline bool isDirectoryMemoryPlace(int index)
	{
		return (index == _directoryMemoryPlace.getIndex());
	}
	
	//! \brief Retrieve the Directory MemoryPlace
	//!
	//! \returns a pointer to the Directory MemoryPlace
	static inline MemoryPlace const *getDirectoryMemoryPlace()
	{
		return &_directoryMemoryPlace;
	}
	
	//! \brief insert a region to the directory
	//!
	//! \param[in] region is the DataAccessRegion to insert
	//! \param[in] homeNode is the home node of the region
	static inline void insert(DataAccessRegion const &region,
			MemoryPlace const *homeNode)
	{
		_homeNodes.insert(region, homeNode);
	}
	
	//! \brief find the home nodes of all the subregions of a region
	//!
	//! This method returns an array of HomeMapEntry elements which
	//! represent the HomeNode for every subregion of the region passed by
	//! the caller of the method. The user of the the HomeNodesArray is
	//! responsible to deallocate it.
	//!
	//! \param[in] region is the DataAccessRegion to lookup
	//!
	//! \returns a vector of HomeMapEntry elements, keeping the HomeNodes
	//! 		of every subregion of region
	static inline HomeNodesArray *find(DataAccessRegion const &region)
	{
		return _homeNodes.find(region);
	}
	
	//! \brief Remove a region from the Directory
	//!
	//! Remove the tracking of a DataAccessRegion inside the Directory.
	//! The DataAccessRegion could be a subregion of a previously
	//! registered region. Only the intersection of that registered region
	//! will be removed
	//!
	//! \param[in] the region to remove from the Directory
	static inline void erase(DataAccessRegion const &region)
	{
		_homeNodes.erase(region);
	}
};

#endif /* DIRECTORY_HPP */
