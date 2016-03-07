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
