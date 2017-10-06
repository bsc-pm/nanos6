/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef LINEAR_REGION_DATA_ACCESS_MAP_HPP
#define LINEAR_REGION_DATA_ACCESS_MAP_HPP

#include "DataAccessRegion.hpp"
#include "LinearRegionMap.hpp"


struct DataAccess;
struct LinearRegionDataAccessMapNode {
	DataAccessRegion _accessRegion;
	DataAccess *_access;
	
	LinearRegionDataAccessMapNode()
		: _accessRegion(), _access(nullptr)
	{
	}
	
	LinearRegionDataAccessMapNode(DataAccessRegion accessRegion)
		: _accessRegion(accessRegion), _access(nullptr)
	{
	}
	
	LinearRegionDataAccessMapNode(DataAccessRegion accessRegion, DataAccess *access)
	: _accessRegion(accessRegion), _access(access)
	{
	}
	
	LinearRegionDataAccessMapNode(LinearRegionDataAccessMapNode const &other)
		: _accessRegion(other._accessRegion), _access(other._access)
	{
	}
	
	DataAccessRegion const &getAccessRegion() const
	{
		return _accessRegion;
	}
	
	DataAccessRegion &getAccessRegion()
	{
		return _accessRegion;
	}
};


class LinearRegionDataAccessMap : public LinearRegionMap<LinearRegionDataAccessMapNode> {
public:
	DataAccess *_superAccess;
	
	LinearRegionDataAccessMap()
		: _superAccess(nullptr)
	{
	}
	
	LinearRegionDataAccessMap(DataAccess *superAccess)
		: _superAccess(superAccess)
	{
	}
};


struct DataAccessNextLinkContents {
	DataAccessRegion _accessRegion;
	DataAccess *_access;
	bool _satisfied;
	
	DataAccessNextLinkContents()
		: _accessRegion(), _access(nullptr), _satisfied(false)
	{
	}
	
	DataAccessNextLinkContents(DataAccessNextLinkContents const &other) = delete;
	
	DataAccessNextLinkContents(DataAccessNextLinkContents &&other)
		: _accessRegion(other._accessRegion), _access(other._access), _satisfied(other._satisfied)
	{
	}
	
	DataAccessNextLinkContents(DataAccessRegion accessRegion, DataAccess *access, bool satisfied)
		: _accessRegion(accessRegion), _access(access), _satisfied(satisfied)
	{
	}
	
	DataAccessRegion const &getAccessRegion() const
	{
		return _accessRegion;
	}
	
	DataAccessRegion &getAccessRegion()
	{
		return _accessRegion;
	}
};


typedef LinearRegionMap<DataAccessNextLinkContents> DataAccessNextLinks;
typedef LinearRegionMap<LinearRegionDataAccessMapNode> DataAccessPreviousLinks;


#include "LinearRegionDataAccessMapImplementation.hpp"


#endif // LINEAR_REGION_DATA_ACCESS_MAP_HPP
