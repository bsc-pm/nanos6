#include "HardwareInfo.hpp"
#include <hwloc.h>
#include <assert.h>

#include "hardware/places/NUMAPlace.hpp"
#include "executors/threads/CPU.hpp"

//! Workaround to deal with changes in different HWLOC versions.
#if HWLOC_API_VERSION < 0x00010b00 
	#define HWLOC_NUMA_ALIAS HWLOC_OBJ_NODE 
#else
	#define HWLOC_NUMA_ALIAS HWLOC_OBJ_NUMANODE
#endif

std::map<int, MemoryPlace*> HardwareInfo::_memoryNodes;
std::map<int, ComputePlace*> HardwareInfo::_computeNodes;
long HardwareInfo::_totalCPUs;

void HardwareInfo::initialize()
{
	_totalCPUs = 0;
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
		//! NUMA node info is available
		for(int i = 0; i < memNodesCount; i++){ 
			//! Create a MemoryPlace for each NUMA node.
			//! Get the hwloc obj representing the NUMA node. 
			hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_NUMA_ALIAS, i);
			//! Create the MemoryPlace representing the NUMA node with its index and AddressSpace. 
			node = new NUMAPlace(obj->logical_index, NUMAAddressSpace);
			//! Add the MemoryPlace to the list of memory nodes of the HardwareInfo.
			_memoryNodes[node->getIndex()] = node;
		}
	} 
	else { 
		//! There is no NUMA info. We assume we have a single MemoryPlace.
		//! Create a MemoryPlace.
		//! TODO: Index is 0 arbitrarily. Maybe a special index should be set.
		//! Create the MemoryPlace representing the NUMA node with its index and AddressSpace. 
		node = new NUMAPlace(/*Index*/0, NUMAAddressSpace);
		//! Add the MemoryPlace to the list of memory nodes of the HardwareInfo.
		_memoryNodes[node->getIndex()] = node;
	}

	//! Get (logical) cores of the machine
	int coresCount = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PU );
	for(int i=0; i<coresCount; i++) {
		hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PU, i);
		hwloc_obj_t nodeNUMA = hwloc_get_ancestor_obj_by_type(topology, HWLOC_NUMA_ALIAS, obj);
		size_t NUMANodeId = nodeNUMA == NULL ? 0 : nodeNUMA->logical_index;
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
