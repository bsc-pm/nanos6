#ifndef LOADER_CPP
#define LOADER_CPP

#include "Loader.hpp"
#include "places/AIOPlace.hpp"
#include "../memory/AIOCache.hpp"
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
    // Load SMP device
    // Create NUMA addressSpace
    AddressSpace * NUMAAddressSpace = new AddressSpace();

    // Get NUMA nodes of the machine
    //int n_nodes = hwloc_get_nbobjs_by_type( _topology, HWLOC_OBJ_NUMANODE );
    int n_nodes = hwloc_get_nbobjs_by_type( _topology, HWLOC_NUMA_ALIAS );

    if(n_nodes != 0){ // NUMA node info is available
        for(int i = 0; i < n_nodes; i++){ //Load nodes
            NUMAPlace *node = createMemoryFromObj(hwloc_get_obj_by_type( _topology, HWLOC_NUMA_ALIAS, i ), NUMAAddressSpace);
            machine->addMemoryNode(node);
        }
    } else { // No NUMA info, fallback on single node
        NUMAPlace *node = createMemoryFromMachine(NUMAAddressSpace); 
        machine->addMemoryNode(node);
    }

    // Get cores of the machine
    int n_cores = hwloc_get_nbobjs_by_type( _topology, HWLOC_OBJ_CORE );

    if(n_cores != 0){ // core info is available
        for(int i = 0; i < n_cores; i++){ //Load nodes
            CPUPlace *node = createCPUFromObj( hwloc_get_obj_by_type( _topology, HWLOC_OBJ_CORE, i ) );
            machine->addComputeNode(node);
        }
    } else { // No core info, fallback detecting processing units
        int n_PUs = hwloc_get_nbobjs_by_type( _topology, HWLOC_OBJ_PU );
        for(int i = 0; i < n_PUs; i++){ //Load nodes
            CPUPlace *node = createCPUFromObj( hwloc_get_obj_by_type( _topology, HWLOC_OBJ_PU, i ) );
            machine->addComputeNode(node);
        }
    }

    // Associate CPUPlaces with NUMAPlaces
    for(Machine::MemoryNodes_t::iterator mem = machine->memoryNodesBegin(); mem != machine->memoryNodesEnd(); ++mem) {
        for(Machine::ComputeNodes_t::iterator pu = machine->computeNodesBegin(); pu != machine->computeNodesEnd(); ++pu) {
            NUMAPlace * numa = (NUMAPlace *) mem->second;
            numa->addPU(pu->second);
            pu->second->addMemoryPlace(mem->second);
        }
    }

    // Load AIO device
    // Create AIO addressSpace
    AddressSpace * AIOAddressSpace = new AddressSpace();
    // Create AIOCache
    AIOCache * AIOcache = new AIOCache();
    // Create AIO Memory Place
    AIOPlace * AIOMemoryPlace = new AIOPlace(/*index*/ -1, AIOcache, AIOAddressSpace);
    machine->addMemoryNode(AIOMemoryPlace);

    // load accelerators
    // load distances

    //other work				
}

NUMAPlace* Loader::createMemoryFromObj(hwloc_obj_t node_obj, AddressSpace * space){
    // Create MemoryPlace representing the NUMA node
    NUMACache * cache = new NUMACache();
    NUMAPlace* node = new NUMAPlace(node_obj->logical_index, cache, space);
    return node;
}

NUMAPlace* Loader::createMemoryFromMachine( AddressSpace * space){
    NUMACache * cache = new NUMACache();
    NUMAPlace* node = new NUMAPlace(0, cache, space); // arbitrary os index // TODO: special index? Negative index?
    return node;
}


#endif //LOADER_CPP
