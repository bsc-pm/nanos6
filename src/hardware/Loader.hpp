#ifndef LOADER_HPP
#define LOADER_HPP

#include <hwloc.h>
#include "Machine.hpp"

class Machine;

class Loader{

private:
	hwloc_topology_t _topology;
	
	MemoryPlace* createMemoryFromObj(hwloc_obj_t node_obj, AddressSpace * space);
	MemoryPlace* createMemoryFromMachine(AddressSpace * space);

	inline ComputePlace* createPUFromObj(hwloc_obj_t pu_obj) {
        	return new ComputePlace( pu_obj->logical_index );
	}
	
public:
	Loader();
	~Loader();
	void load(Machine *machine);
};


#endif //LOADER_HPP
