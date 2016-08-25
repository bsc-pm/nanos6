#ifndef LOADER_CPP
#define LOADER_CPP

#include <hwloc.h>

#include "hardware/places/MemoryPlace.hpp"
#include "hardware/places/ComputePlace.hpp"
#include "hardware/Machine.hpp"


Loader::Loader(){
	hwloc_topology_init(&topology);  // initialization
  	hwloc_topology_load(topology);   // actual detection 
}

Loader::~Loader(){
	hwloc_topology_destroy(topology); // release resources
}

void Loader::load(Machine *machine){
	int n_nodes = hwloc_get_nbobjs_by_type( topology, HWLOC_OBJ_NUMANODE );
		
	if(n_nodes != 0){ // NUMA node info is available
		for(int i = 0; i < n_nodes; i++){ //Load nodes
			MemoryPlace *node = create_memory_from_obj( hwloc_get_obj_by_type( topology, HWLOC_OBJ_NUMANODE, i ) );
			machine->_addNode(node);
		}
	} else { // No NUMA info, fallback on single node
		MemoryPlace *node = create_memory_from_machine(); 
		machine->_addNode(node);
	}
 		
	// load accelerators
		// load distances
	
	//other work				
}


#endif //LOADER_CPP
