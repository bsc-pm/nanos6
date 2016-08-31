#ifndef MEMORY_REGION_CPP
#define MEMORY_REGION_CPP

#include <numa.h>
#include <numaif.h>

#include "MemoryRegion.hpp"
#include "hardware/Machine.hpp"

/*
void MemoryRegion::merge(MemoryRegion *other){
	
	char *my_start = (char *) _address;	
	char *other_start = (char *) other->_address;
	char *my_end = my_start + _size;
	char *other_end = other_start + other->_size;

	// check if there is an intersection
	if(( my_start > other_start && my_end < other_end ) ||
	   ( my_start < other_start && my_end > other_end ) ||
	   ( my_end < other_start) || (other_end < my_start) ){
		return; // improve this
	}
	
	if(my_start < other_start && my_end < other_end){ 	// intersection [ (] ) []=this ()=other
		_size += other_end - my_end;   
	} else { 						// intersection (  [)   ]  []=this  ()=other
		_address = other->_address;
	}	
}

int MemoryRegion::locate(void){
	if(_present) return 0; // already located	
	
	long pagesize = Machine::instance()->getPageSize();
	void *ptr = (void*)( ( long )_address & ~(pagesize-1) );
	int npages = (( (long)_address - (long)ptr ) + _size) / pagesize;

	// allocate support structures for move_pages
	void **pages = new void*[npages];
	int *status = new int[npages];
	
	// set page addresses
	for(int i = 0; i < npages; i++){
		pages[i] = ptr + i*pagesize;
	}

	move_pages(0, npages, pages, NULL, status, 0);
	
	// check status
	std::vector<int> nodes;
	bool interleaved = true;

	 *
	 * Check status of the region
	 * If interleaved, on the first round nodes will be added to the vector and on the rest checked for the interleave pattern to be respected
	 * If there is only a single node, on the first round it will add it to the vector and on the rest pointless checking (overhead)
	 * 
	 * All elements are checked in case of fringe distributions, all pages on one node except the last one, or a more plausible case,
	 *   X pages per node equally distributed (is this possible?)
	 * 
	 
	for(int i = 0; i < npages; i++){
		int node = status[i];
		if(node = ENOENT){
			return ENOENT; // error, pages not present in memory (may be untouched)
		}
		if(std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
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
		_location[i] = Machine::instance()->getNode(nodes[i]);
	}
	
	_present = true;
	return 0; // all Ok
}
*/

#endif //MEMORY_REGION_CPP
