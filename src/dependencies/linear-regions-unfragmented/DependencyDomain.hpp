/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEPENDENCY_DOMAIN_HPP
#define DEPENDENCY_DOMAIN_HPP


#include "LinearRegionDataAccessMap.hpp"
#include "lowlevel/SpinLock.hpp"


struct DependencyDomain {
	LinearRegionDataAccessMap _map;
	SpinLock _lock;
	
	DependencyDomain()
		: _map(), _lock()
	{
	}
};


#endif // DEPENDENCY_DOMAIN_HPP
