#ifndef LOADER_HPP
#define LOADER_HPP

#include <hwloc.h>
#include "Machine.hpp"
#include "places/CPUPlace.hpp"
#include "places/NUMAPlace.hpp"

class Machine;

class Loader{

private:
	hwloc_topology_t _topology;
	
	NUMAPlace* createMemoryFromObj(hwloc_obj_t node_obj, AddressSpace * space);
	NUMAPlace* createMemoryFromMachine(AddressSpace * space);

	inline CPUPlace* createCPUFromObj(hwloc_obj_t cpu_obj) {
        	return new CPUPlace( cpu_obj->logical_index );
	}
	
public:
	Loader();
	~Loader();
	void load(Machine *machine);
};


#endif //LOADER_HPP
