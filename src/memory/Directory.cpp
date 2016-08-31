#ifndef DIRECTORY_CPP
#define DIRECTORY_CPP

#include "Directory.hpp"

bool Directory::empty(){
	return _set.empty();
}

int Directory::size(){
	return _set.size();
}

typename Directory::iterator Directory::begin(){
	return _set.begin(); 
}

typename Directory::iterator Directory::end(){
	return _set.end();
}

typename Directory::iterator Directory::find(void *address){
	return _set.find(address);
}

void Directory::insert(void *baseAddress){
	// TODO check if contains other objects. (engulf them in the _homes set)
	// TODO check if contained by other. (send to _homes of the object)
	MemoryObject *obj = new MemoryObject(baseAddress);
	_set.insert( *obj );
}

typename Directory::iterator Directory::erase(typename Directory::iterator position){
	MemoryObject* obj = &(*position);
	Directory::iterator it = _set.erase(position);
	delete obj;
	return it;
}
/*
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

*/

#endif //DIRECTORY_CPP
