#ifndef LOADER_CPP
#define LOADER_CPP

#include "Loader.hpp"
#if HWLOC_API_VERSION < 0x00010b00 
    #define HWLOC_NUMA_ALIAS HWLOC_OBJ_NODE 
#else
    #define HWLOC_NUMA_ALIAS HWLOC_OBJ_NUMANODE
#endif

Loader::Loader(){
    hwloc_topology_init(&_topology);  // initialization
    hwloc_topology_load(_topology);   // actual detection 
}

Loader::~Loader(){
    hwloc_topology_destroy(_topology); // release resources
}

void Loader::load(Machine *machine){
    // Create NUMA addressSpace
    AddressSpace * numaAddressSpace = new AddressSpace(machine->getNewAddressSpaceId());

    // Get NUMA nodes of the machine
    //int n_nodes = hwloc_get_nbobjs_by_type( _topology, HWLOC_OBJ_NUMANODE );
    int n_nodes = hwloc_get_nbobjs_by_type( _topology, HWLOC_NUMA_ALIAS );

    if(n_nodes != 0){ // NUMA node info is available
        for(int i = 0; i < n_nodes; i++){ //Load nodes
            //MemoryPlace *node = createMemoryFromObj(hwloc_get_obj_by_type( _topology, HWLOC_OBJ_NUMANODE, i ), numaAddressSpace);
            MemoryPlace *node = createMemoryFromObj(hwloc_get_obj_by_type( _topology, HWLOC_NUMA_ALIAS, i ), numaAddressSpace);
            machine->addMemoryNode(node);
        }
    } else { // No NUMA info, fallback on single node
        MemoryPlace *node = createMemoryFromMachine(numaAddressSpace); 
        machine->addMemoryNode(node);
    }

    // Get cores of the machine
    int n_cores = hwloc_get_nbobjs_by_type( _topology, HWLOC_OBJ_CORE );

    if(n_cores != 0){ // core info is available
        for(int i = 0; i < n_cores; i++){ //Load nodes
            ComputePlace *node = createPUFromObj( hwloc_get_obj_by_type( _topology, HWLOC_OBJ_CORE, i ) );
            machine->addComputeNode(node);
        }
    } else { // No core info, fallback detecting processing units
        int n_PUs = hwloc_get_nbobjs_by_type( _topology, HWLOC_OBJ_PU );
        for(int i = 0; i < n_cores; i++){ //Load nodes
            ComputePlace *node = createPUFromObj( hwloc_get_obj_by_type( _topology, HWLOC_OBJ_PU, i ) );
            machine->addComputeNode(node);
        }
    }

    // Associate ComputePlaces with MemoryPlaces
    for(Machine::MemoryNodes_t::iterator mem = machine->memoryNodesBegin(); mem != machine->memoryNodesEnd(); ++mem) {
        for(Machine::ComputeNodes_t::iterator pu = machine->computeNodesBegin(); pu != machine->computeNodesEnd(); ++pu) {
            mem->second->addPU(pu->second);
            pu->second->addMemoryPlace(mem->second);
        }
    }

    // load accelerators
    // load distances

    //other work				
}

MemoryPlace* Loader::createMemoryFromObj(hwloc_obj_t node_obj, AddressSpace * space){
    // Create MemoryPlace representing the NUMA node
    MemoryPlace* node = new MemoryPlace(node_obj->logical_index, space);
    return node;
}

MemoryPlace* Loader::createMemoryFromMachine( AddressSpace * space){
    MemoryPlace* node = new MemoryPlace(0, space); // arbitrary os index // TODO: special index? Negative index?
    return node;
}


#endif //LOADER_CPP

