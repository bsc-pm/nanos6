/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef LINEAR_REGION_DATA_ACCESS_MAP_HPP
#define LINEAR_REGION_DATA_ACCESS_MAP_HPP

#include "DataAccessRange.hpp"
#include "LinearRegionMap.hpp"


struct DataAccess;
struct LinearRegionDataAccessMapNode {
	DataAccessRange _accessRange;
	DataAccess *_access;
	
	LinearRegionDataAccessMapNode()
		: _accessRange(), _access(nullptr)
	{
	}
	
	LinearRegionDataAccessMapNode(DataAccessRange accessRange)
		: _accessRange(accessRange), _access(nullptr)
	{
	}
	
	LinearRegionDataAccessMapNode(DataAccessRange accessRange, DataAccess *access)
	: _accessRange(accessRange), _access(access)
	{
	}
	
	LinearRegionDataAccessMapNode(LinearRegionDataAccessMapNode const &other)
		: _accessRange(other._accessRange), _access(other._access)
	{
	}
	
	DataAccessRange const &getAccessRange() const
	{
		return _accessRange;
	}
	
	DataAccessRange &getAccessRange()
	{
		return _accessRange;
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
	DataAccessRange _accessRange;
	DataAccess *_access;
	bool _satisfied;
	
	DataAccessNextLinkContents()
		: _accessRange(), _access(nullptr), _satisfied(false)
	{
	}
	
	DataAccessNextLinkContents(DataAccessNextLinkContents const &other) = delete;
	
	DataAccessNextLinkContents(DataAccessNextLinkContents &&other)
		: _accessRange(other._accessRange), _access(other._access), _satisfied(other._satisfied)
	{
	}
	
	DataAccessNextLinkContents(DataAccessRange accessRange, DataAccess *access, bool satisfied)
		: _accessRange(accessRange), _access(access), _satisfied(satisfied)
	{
	}
	
	DataAccessRange const &getAccessRange() const
	{
		return _accessRange;
	}
	
	DataAccessRange &getAccessRange()
	{
		return _accessRange;
	}
};


typedef LinearRegionMap<DataAccessNextLinkContents> DataAccessNextLinks;
typedef LinearRegionMap<LinearRegionDataAccessMapNode> DataAccessPreviousLinks;


#include "LinearRegionDataAccessMapImplementation.hpp"


#endif // LINEAR_REGION_DATA_ACCESS_MAP_HPP
