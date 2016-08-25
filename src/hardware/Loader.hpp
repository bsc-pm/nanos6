#ifndef LOADER_HPP
#define LOADER_HPP

#include <hwloc.h>

#include "hardware/places/MemoryPlace.hpp"
#include "hardware/places/ComputePlace.hpp"

class Machine;

class Loader{

private:
	hwloc_topology_t topology;
	
	MemoryPlace* create_memory_from_obj(hwloc_obj_t node_obj){
        	hwloc_bitmap_t cpuset = node_obj->cpuset;

        	MemoryPlace* node = new MemoryPlace(node_obj->os_index);

        	if(!hwloc_bitmap_iszero(cpuset) && !hwloc_bitmap_isfull(cpuset)){ // no cpus in the node

        	        int index = -1; // next() with -1 = first element
        	        for(unsigned i = 0, len = hwloc_bitmap_weight(cpuset); i < len; i++){
        	                index = hwloc_bitmap_next( cpuset, index );
                	        hwloc_obj_t cpu_obj = hwloc_get_pu_obj_by_os_index( topology, index );
                        	node->_addPU( create_cpu_from_obj( cpu_obj, node ) );
               	 	}
        	}

       	 	return node;
	}

	MemoryPlace* create_memory_from_machine(){
        	MemoryPlace* node = new MemoryPlace(0); // arbitrary os index // TODO: special index?

        	int n_cpus = hwloc_get_nbobjs_by_type( topology, HWLOC_OBJ_PU );
        	for(int i = 0; i < n_cpus; i++){
                	hwloc_obj_t cpu_obj = hwloc_get_obj_by_type( topology, HWLOC_OBJ_NUMANODE, i );
                	node->_addPU( create_cpu_from_obj( cpu_obj, node ) );
        	}
		return node;
	}

	ComputePlace* create_cpu_from_obj(hwloc_obj_t cpu_obj, MemoryPlace *node){
        	return new ComputePlace( cpu_obj->os_index, node );
	}

	
public:
	Loader();
	~Loader();
	void load(Machine *machine);
		
};


#endif //LOADER_HPP
