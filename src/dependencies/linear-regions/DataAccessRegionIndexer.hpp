/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_REGION_INDEXER_HPP
#define DATA_ACCESS_REGION_INDEXER_HPP


#include "LinearRegionMap.hpp"
#include "LinearRegionMapImplementation.hpp"


template <typename ContentType>
using DataAccessRegionIndexer = LinearRegionMap<ContentType>;


#endif // DATA_ACCESS_REGION_INDEXER_HPP
