/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_REGION_INDEXER_HPP
#define DATA_ACCESS_REGION_INDEXER_HPP


#include "DiscreteAddressMap.hpp"


template <typename ContentType>
using DataAccessRegionIndexer = DiscreteAddressMap<ContentType>;


#endif // DATA_ACCESS_REGION_INDEXER_HPP
