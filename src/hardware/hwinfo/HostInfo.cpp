/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "HostInfo.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include <hwloc.h>
#include <assert.h>
#include <unistd.h>

#include "hardware/places/NUMAPlace.hpp"
#include "executors/threads/CPU.hpp"

//! Workaround to deal with changes in different HWLOC versions.
#if HWLOC_API_VERSION < 0x00010b00 
	#define HWLOC_NUMA_ALIAS HWLOC_OBJ_NODE 
#else
	#define HWLOC_NUMA_ALIAS HWLOC_OBJ_NUMANODE
#endif

#define DEFAULT_CACHE_LINE_SIZE 64

void HostInfo::initialize()
{
//! Check that hwloc headers match with runtime.
#if HWLOC_API_VERSION >= 0x00020000 
	FatalErrorHandler::failIf(hwloc_get_api_version() < 0x20000, "hwloc headers are more recent than runtime library.");
#else
	FatalErrorHandler::failIf(hwloc_get_api_version() >= 0x20000, "hwloc headers are older than runtime library.");
#endif
	//! Hardware discovery
	hwloc_topology_t topology;
	hwloc_topology_init(&topology);  // initialization
	hwloc_topology_load(topology);   // actual detection 

	//! Create NUMA addressSpace
	AddressSpace * NUMAAddressSpace = new AddressSpace();

	//! Get NUMA nodes of the machine.
	//! NUMA node means: A set of processors around memory which the processors can directly access. (Extracted from hwloc documentation)
	int memNodesCount = hwloc_get_nbobjs_by_type( topology, HWLOC_NUMA_ALIAS );

	//! Check if HWLOC has found any NUMA node.
	NUMAPlace * node = nullptr;
	if(memNodesCount != 0){ 
		_memoryPlaces.resize(memNodesCount);
		//! NUMA node info is available
		for(int i = 0; i < memNodesCount; i++){ 
			//! Create a MemoryPlace for each NUMA node.
			//! Get the hwloc obj representing the NUMA node. 
			hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_NUMA_ALIAS, i);
			assert(obj != nullptr);
			//! Create the MemoryPlace representing the NUMA node with its index and AddressSpace. 
			node = new NUMAPlace(obj->logical_index, NUMAAddressSpace);
			//! Add the MemoryPlace to the list of memory nodes of the HardwareInfo.
			_memoryPlaces[node->getIndex()] = node;
			node = nullptr;
		}
	} 
	else {
		_memoryPlaces.resize(1);
		//! There is no NUMA info. We assume we have a single MemoryPlace.
		//! Create a MemoryPlace.
		//! TODO: Index is 0 arbitrarily. Maybe a special index should be set.
		//! Create the MemoryPlace representing the NUMA node with its index and AddressSpace. 
		node = new NUMAPlace(/*Index*/0, NUMAAddressSpace);
		//! Add the MemoryPlace to the list of memory nodes of the HardwareInfo.
		_memoryPlaces[node->getIndex()] = node;
	}

	//! Get (logical) cores of the machine
	int coresCount = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU);
	_computePlaces.resize(coresCount);
	for(int i = 0; i < coresCount; i++) {
		hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
		assert(obj != nullptr);
#if HWLOC_API_VERSION >= 0x00020000
		hwloc_obj_t ancestor = nullptr; 
		hwloc_obj_t nodeNUMA = nullptr;

		//! NUMA node can be found in different depths of ancestors (ordered from deeper to narrower): 
		//! 1. A L3CACHE object / A GROUP object.
		//! 2. The most common is a PACKAGE object.
		//! 3. The MACHINE object.
		//! ref: https://www.open-mpi.org/projects/hwloc/doc/v2.0.0/a00327.php
		ancestor = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_L3CACHE, obj);
		if(ancestor == nullptr || ancestor->memory_arity != 1) {
			ancestor = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_GROUP, obj);
			if(ancestor == nullptr || ancestor->memory_arity != 1) {
				ancestor = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_PACKAGE, obj);
				if(ancestor == nullptr || ancestor->memory_arity != 1) {
					ancestor = hwloc_get_ancestor_obj_by_type(topology, HWLOC_OBJ_MACHINE, obj);
				}
			}
		}
		assert(ancestor != nullptr);
		assert(ancestor->memory_arity == 1);

		nodeNUMA = ancestor->memory_first_child;
		assert(nodeNUMA != nullptr);
		assert(hwloc_obj_type_is_memory(nodeNUMA->type));
#else
		hwloc_obj_t nodeNUMA = hwloc_get_ancestor_obj_by_type(topology, HWLOC_NUMA_ALIAS, obj);
#endif
		size_t NUMANodeId = nodeNUMA == NULL ? 0 : nodeNUMA->logical_index;
		CPU * cpu = new CPU( /*systemCPUID*/ obj->os_index, /*virtualCPUID*/ obj->logical_index, NUMANodeId);
		_computePlaces[obj->logical_index] = cpu;
	}
	
	hwloc_obj_t cache = nullptr;
#if HWLOC_API_VERSION >= 0x00020000
	cache = hwloc_get_obj_by_type(topology, HWLOC_OBJ_L1CACHE, 0);
#else
	int cacheDepth = hwloc_get_cache_type_depth(topology, 1, HWLOC_OBJ_CACHE_DATA);
	
	if (cacheDepth != HWLOC_TYPE_DEPTH_MULTIPLE && cacheDepth != HWLOC_TYPE_DEPTH_UNKNOWN) {
		cache = hwloc_get_obj_by_depth(topology, cacheDepth, 0);
	}
#endif

	if ((cache != nullptr) && (cache->attr->cache.linesize != 0)) {
		_cacheLineSize = cache->attr->cache.linesize;
	} else {
		// In some machines, such as HCA-Merlin or Dibona,
		// hwloc cannot obtain cache information or just returns 0
		_cacheLineSize = DEFAULT_CACHE_LINE_SIZE;
	}
	
	//! Attributes of system's memory
	_pageSize = sysconf(_SC_PAGESIZE);
	//! This is not so portable, but it works for more Unix-like stuff
	size_t nrPhysicalPages = sysconf(_SC_PHYS_PAGES);
	_physicalMemorySize = nrPhysicalPages * _pageSize;
	
	//! Associate CPUs with NUMA nodes
	for(std::vector<MemoryPlace *>::iterator numaNode = _memoryPlaces.begin(); numaNode != _memoryPlaces.end(); ++numaNode) {
		for(std::vector<ComputePlace *>::iterator cpu = _computePlaces.begin(); cpu != _computePlaces.end(); ++cpu) {
			((NUMAPlace*)*numaNode)->addComputePlace(*cpu);
			(*cpu)->addMemoryPlace(*numaNode);
		}
	}

	//other work				
	hwloc_topology_destroy(topology); // release resources
}

void HostInfo::shutdown()
{
	for (size_t i = 0; i < _memoryPlaces.size(); ++i) {
		delete _memoryPlaces[i];
	}
	for (size_t i = 0; i < _computePlaces.size(); ++i) {
		delete _computePlaces[i];
	}
}
