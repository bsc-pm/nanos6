#ifndef DIRECTORY_HPP
#define DIRECTORY_HPP

#include "MemoryRegion.hpp"
#include "../dependencies/linear-regions-unfragmented/LinearRegionMap.hpp" 

class Directory {
private:
	typedef LinearRegionMap<MemoryRegion*> regions_t;	
	
	regions_t _map;
	
	static Directory *_instance;
	
	Directory(){
		
	}

public:
	
	
	bool insert(void *addr, size_t size, bool interleaved = false, int nLocations = 0, bool present = false, MemoryPlace *location = nullptr);	
	
	// singleton methods
	static void init(){
		_instance = new Directory();
	}

	static void destroy(){
		delete _instance;
	}
	
}



#endif //DIRECTORY_HPP
