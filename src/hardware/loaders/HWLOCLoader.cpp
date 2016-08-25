#ifndef HWLOC_LOADER_CPP
#define HWLOC_LOADER_CPP

#include "Loader.hpp"
#include <hwloc.h>

class HWLOCLoader : public Loader {

private:
	typedef *ComputePlace processors_t;
	typedef *MemoryPlace memories_t;	

	hwloc_topology_t topology;
	
	MemoryPlace* create_memory_from_obj(hwloc_obj_t node_obj){
		hwloc_bitmap_t cpuset = node_obj->cpuset;

		MemoryPlace* node = new MemoryPlace(node_obj->os_index);

		if(!hwloc_bitmap_iszero(cpuset) && !hwloc_bitmap_isfull(cpuset)) // no cpus in the node
	
			int index = -1; // next() with -1 = first element
			for(unsigned i = 0, size_t len = hwloc_bitmap_weight(cpuset); i < len; i++){
				index = hwloc_bitmap_next( cpuset, index );
				hwloc_obj_t cpu_obj = hwloc_get_pu_obj_by_os_index( topology, index );
				node->_addPU( create_cpu_from_obj( cpu_obj ) ); 
			}
		}

		return node;
	}

	MemoryPlace* create_memory_from_machine(){
		MemoryPlace* node = new MemoryPlace(0); // arbitrary os index // TODO: special index?
		
		int n_cpus = hwloc_get_nbobjs_by_type( topology, HWLOC_OBJ_PU );
		for(unsigned i = 0; i < n_cpus; i++){
			hwloc_obj_t cpu_obj = hwloc_get_obj_by_type( topology, HWLOC_OBJ_NUMANODE, i );
			node->_addPU( cpu_obj, node );
		}
	}

	ComputePlace* create_cpu_from_obj(hwloc_obj_t cpu_obj, MemoryPlace *node){
		return new ComputePlace( cpu_obj->os_index, node );
	}

public:
	HWLOCLoader(){
		hwloc_topology_init(&topology);  // initialization
  		hwloc_topology_load(topology);   // actual detection 
	}

	HWLOCLoader~(){
		hwloc_topology_destroy(topology); // release resources
	}


	void load(Machine *machine){
		
		// load the numa nodes
		int n_nodes = hwloc_get_nbobjs_by_type( topology, HWLOC_OBJ_NUMANODE );
		
		if(n_nodes != 0){ // NUMA node info is available
			for(unsigned i = 0; i < n_nodes; i++){ //Load nodes
				MemoryPlace *node = create_memory_from_obj( hwloc_get_obj_by_type( topology, HWLOC_OBJ_NUMANODE, i ) );
				machine->_addNode(node);
			}
		} else { // No NUMA info, fallback on single node
			machine->_addNode(node);
		}
 		
		// load accelerators

		// load distances
		
		//other work		
	}
		
}


#endif //HWLOC_LOADER_CPP
