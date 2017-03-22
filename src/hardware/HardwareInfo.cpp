#include "HardwareInfo.hpp"
#include <hwloc.h>
#include <assert.h>

#include "hardware/places/NUMAPlace.hpp"
#include "memory/cache/NUMACache.hpp"
#include "executors/threads/CPU.hpp"
#include "memory/directory/Directory.hpp"

//! Workaround to deal with changes in different HWLOC versions.
#if HWLOC_API_VERSION < 0x00010b00 
    #define HWLOC_NUMA_ALIAS HWLOC_OBJ_NODE 
#else
    #define HWLOC_NUMA_ALIAS HWLOC_OBJ_NUMANODE
#endif

std::map<int, MemoryPlace*> HardwareInfo::_memoryNodes;
std::map<int, ComputePlace*> HardwareInfo::_computeNodes;
/*std::atomic<long>*/ long HardwareInfo::_totalCPUs;
//Distances_t _distances;
long HardwareInfo::_pageSize;
std::size_t HardwareInfo::_lastLevelCacheSize;
std::size_t HardwareInfo::_lastLevelCacheLineSize;

void HardwareInfo::initialize()
{
    _totalCPUs = 0;
    _pageSize = getpagesize();
    _lastLevelCacheSize = 0;
    _lastLevelCacheLineSize = 0;
    //! Hardware discovery
	hwloc_topology_t topology;
    hwloc_topology_init(&topology);  // initialization
    hwloc_topology_load(topology);   // actual detection 

    //! Start wih NUMA devices.
    //! TODO: We assume we are always in NUMA. Is it okay?

    //! Create NUMA addressSpace
    AddressSpace * NUMAAddressSpace = new AddressSpace();

    //! Get NUMA nodes of the machine.
    //! NUMA node means: A set of processors around memory which the processors can directly access. (Extracted from hwloc documentation)
    int memNodesCount = hwloc_get_nbobjs_by_type( topology, HWLOC_NUMA_ALIAS );

    //! Check if HWLOC has found any NUMA node.
    NUMACache * cache = nullptr;
    NUMAPlace * node = nullptr;
    if(memNodesCount != 0){ 
        //! NUMA node info is available
        //! Create a ReadyQueue in the scheduler for each NUMA node.
        Scheduler::createReadyQueues(memNodesCount);
        //! Create lastLevelCacheTracking for each NUMA node.
        Directory::createLastLevelCacheTracking(memNodesCount);
        for(int i = 0; i < memNodesCount; i++){ 
            //! Create a MemoryPlace for each NUMA node.
            //! Get the hwloc obj representing the NUMA node. 
            hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_NUMA_ALIAS, i);
            //! Create a Cache for the MemoryPlace. Assign the same index than the MemoryPlace.
            cache = new NUMACache(obj->logical_index);
            assert(cache != nullptr && "No cache has been created");
            //! Create the MemoryPlace representing the NUMA node with its index, cache and AddressSpace. 
            node = new NUMAPlace(cache->getIndex(), cache, NUMAAddressSpace);
            //! Add the MemoryPlace to the list of memory nodes of the HardwareInfo.
            _memoryNodes[node->getIndex()] = node;
        }
    } 
    else { 
        //! There is no NUMA info. We assume we have a single MemoryPlace.
        //! Create a Cache for the MemoryPlace. Assign the same index than the MemoryPlace.
        //! TODO: Index is 0 arbitrarily. Maybe a special index should be set.
        cache = new NUMACache(/*index*/0);
        //! Create a ReadyQueue in the scheduler for each NUMA node.
        Scheduler::createReadyQueues(1);
        //! Create lastLevelCacheTracking for each NUMA node.
        Directory::createLastLevelCacheTracking(1);
        assert(cache != nullptr && "No cache has been created");
        //! Create the MemoryPlace representing the NUMA node with its index, cache and AddressSpace. 
        node = new NUMAPlace(cache->getIndex(), cache, NUMAAddressSpace);
        //! Add the MemoryPlace to the list of memory nodes of the HardwareInfo.
        _memoryNodes[node->getIndex()] = node;
    }

    //! Get last level cache size of the machine and its line size.
    hwloc_obj_t root = hwloc_get_root_obj(topology);
    hwloc_obj_t obj = root->children[0];
    //while(true) {
    //    if(obj->type == HWLOC_OBJ_CACHE) {
    //        if(obj->attr->type = HWLOC_OBJ_CACHE_UNIFIED && obj->attr->depth == 3) 
    //            break;
    //    }
    //    obj = obj->children[0];
    //}
    while(!(obj->type == HWLOC_OBJ_CACHE && obj->attr->cache.type == HWLOC_OBJ_CACHE_UNIFIED/* && obj->attr->cache.depth == 3*/)) {
        obj = obj->children[0];
    }
    _lastLevelCacheSize = obj->attr->cache.size;
    _lastLevelCacheLineSize = obj->attr->cache.linesize;


    //! Get (logical) cores of the machine
    int coresCount = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU );
    for(int i=0; i<coresCount; i++) {
        hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
        hwloc_obj_t nodeNUMA = hwloc_get_ancestor_obj_by_type(topology, HWLOC_NUMA_ALIAS, obj);
        size_t NUMANodeId = nodeNUMA == NULL ? 0 : nodeNUMA->logical_index;
        //! TODO: Create an atomic global counter to assign CPUs
        CPU * cpu = new CPU( /*systemCPUID*/ obj->logical_index, /*virtualCPUID*/ _totalCPUs++, NUMANodeId);
        _computeNodes[obj->logical_index] = cpu;
    }

    //! Associate CPUs with NUMA nodes
    for(MemoryNodes_t::iterator numaNode = _memoryNodes.begin(); numaNode != _memoryNodes.end(); ++numaNode) {
        for(ComputeNodes_t::iterator cpu = _computeNodes.begin(); cpu != _computeNodes.end(); ++cpu) {
            ((NUMAPlace*) (numaNode->second))->addComputePlace(cpu->second);
            cpu->second->addMemoryPlace(numaNode->second);
        }
    }

    //! Load AIO device
    //! Create AIO addressSpace
    //AddressSpace * AIOAddressSpace = new AddressSpace();
    //! Create AIOCache
    //AIOCache * AIOcache = new AIOCache(/*index*/-1);
    //! Create AIO Memory Place
    //AIOPlace * AIOMemoryPlace = new AIOPlace(/*index*/ -1, AIOcache, AIOAddressSpace);
    //_memoryNodes(AIOMemoryPlace);

    // load accelerators
    // load distances

    //other work				
    hwloc_topology_destroy(topology); // release resources
}

std::vector<int> HardwareInfo::getComputeNodeIndexes(){
    //! Create a new vector with the correct size. This automatically initialize all the positions to a value.
	std::vector<int> indexes(_computeNodes.size());

    //! Double iterator needed to overwrite the already initialized positions of the vector.
    int i = 0;
	for(ComputeNodes_t::iterator it = _computeNodes.begin(); 
        it != _computeNodes.end(); 
        ++it, ++i)
    {
        indexes[i] = it->first;
	}

	return indexes;
}

std::vector<int> HardwareInfo::getMemoryNodeIndexes(){
    //! Create a new vector with the correct size. This automatically initialize all the positions to a value.
	std::vector<int> indexes(_memoryNodes.size());

    //! Double iterator needed to overwrite the already initialized positions of the vector.
    int i = 0;
	for(MemoryNodes_t::iterator it = _memoryNodes.begin(); 
        it != _memoryNodes.end(); 
        ++it, ++i)
    {
        indexes[i] = it->first;
	}

	return indexes;
}

std::vector<ComputePlace*> HardwareInfo::getComputeNodes(){
    //! Create a new vector with the correct size. This automatically initialize all the positions to a value.
	std::vector<ComputePlace*> nodes(_computeNodes.size());

    //! Double iterator needed to overwrite the already initialized positions of the vector.
    int i = 0;
	for(ComputeNodes_t::iterator it = _computeNodes.begin(); 
        it != _computeNodes.end(); 
        ++it, ++i)
    {
        nodes[i] = it->second;
	}

	return nodes;
}

std::vector<MemoryPlace*> HardwareInfo::getMemoryNodes(){
    //! Create a new vector with the correct size. This automatically initialize all the positions to a value.
	std::vector<MemoryPlace*> nodes(_memoryNodes.size());

    //! Double iterator needed to overwrite the already initialized positions of the vector.
    int i = 0;
	for(MemoryNodes_t::iterator it = _memoryNodes.begin(); 
        it != _memoryNodes.end(); 
        ++it, ++i)
    {
        nodes[i] = it->second;
	}

	return nodes;
}
