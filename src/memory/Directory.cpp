#ifndef DIRECTORY_CPP
#define DIRECTORY_CPP

#include "Directory.hpp"
#include "../dependencies/linear-regions-unfragmentes/LinearRegionMap.hpp"

bool Directory::insert(void *addr, size_t size, bool interleaved, int nNodes, bool present, int** nodes){
	
	// create region object
	MemoryRegion *region = new MemoryRegion(addr, size, interleaved, nNodes, present, nodes);

	// merge with intersecting regions when not interleaved and present
	if(!interleaved && present){ 
		
		_map.processIntersecting( 
			region->_range,
			[&](LinearRegionMap::iterator position) -> bool {i
				region.merge(position); 
				map->erase(position); 
				return true;
			}
		);
	}

	// insert to directory
	_map.insert(region); 
	return true;
}

#endif //DIRECTORY_CPP
