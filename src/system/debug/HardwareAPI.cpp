#include "hardware/Machine.hpp"
#include "hardware/places/MemoryPlace.hpp"
#include "hardware/places/ComputePlace.hpp"

void nanos_initialize_hardware(void){
	Machine.create();
}

long nanos_get_process_pagesize(void){
	return Machine.getpagesize();
}

int nanos_get_node_count(void){
	return Machine.getNodeCount();
}

int nanos_get_cpu_count(void){
	int count = 0;
	for(nodes_t::iterator it = Machine.nodesBegin(); it != Machine.nodesEnd(); ++it){
		count += it->second->getCPUCount();
	} 

	return count;
}

int nanos_get_cpu_count_by_node(int index){
	return Machine.getNode(index)->getCPUCount();
}

std::vector<int>* nanos_get_nodes(void){
	return Machine.getNodeIndexes();
}

std::vector<int>* nanos_get_cpus_by_node(int index){
	return Machine.getNode(index)->getCPUIndexes();
}
