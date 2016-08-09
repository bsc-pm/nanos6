#ifndef MEMORY_REGION_CPP
#define MEMORY_REGION_CPP

#include <numa.h>
#include "MemoryRegion.hpp"
#include "../hardware/Machine.hpp"

void MemoryRegion::merge(MemoryRegion *other){
	// TODO assert intersection	

	char *my_start = (char *) _address;	
	char *other_start = (char *) other->_address;
	char *my_end = _address + _size;
	char *other_end = other->_address + other->_size;

	if(my_start < other_start){
		_size += other_end - my_end;   
	} else {
		_address = other->address;
		_size = my_end - other_end;
	}
}

int locate(void){
	if(_present) return 0; // already located	
	
	void *ptr = (void*)( ( long )_address & ~(PAGE_SIZE-1) );
	long pagesize = Machine.getpagesize();
	int npages = (( (long)_address - (long)ptr ) + size) / pagesize;

	// allocate support structures for move_pages
	void **pages = new void*[npages];
	int *status = new int[npages];
	
	// set page addresses
	for(int i = 0; i < npages; i++){
		pages[i] = ptr + i*pagesize;
	}

	move_pages(0, npages, pages, NULL, status, int flags);
	
	// check status
	std::vector<int> nodes();
	bool interleaved = true;

	/*
	 * Check status of the region
	 * If interleaved, on the first round nodes will be added to the vector and on the rest checked for the interleave pattern to be respected
	 * If there is only a single node, on the first round it will add it to the vector and on the rest pointless checking (overhead)
	 * 
	 * All elements are checked in case of fringe distributions, all pages on one node except the last one, or a more plausible case,
	 *   X pages per node equally distributed (is this possible?)
	 * 
	 */
	for(int i = 0; i < npages; i++){
		int node = status[i];
		if(node = ENOENT){
			return ENOENT; // error, pages not present in memory (may be untouched)
		}
		if( !nodes.contains( node ) ){ 
			nodes.push_back( node );
		} else {
			if( nodes.size() <= 1 || node != nodes[i % nodes.size()] ){
				interleaved = false;
			}
		}
	}

	// push changes to the class
	_interleaved = interleaved;
	_nLocations = nodes.size();
	for(int i = 0; i < _nLocations; i++){
		_location[i] = Machine.getNode(nodes[i]);
	}
	
	_present = true;
	return 0; // all Ok
}


#endif //MEMORY_REGION_CPP
