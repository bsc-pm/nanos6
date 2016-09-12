#ifndef MACHINE_CPP
#define MACHINE_CPP

#include "Machine.hpp"
#include "Loader.hpp"

#include <assert.h>

Machine *Machine::_machine = nullptr;

void Machine::initialize()
{
    assert( _machine == nullptr );
    _machine = new Machine();
    Loader l;
    l.load(_machine);
    _machine->_pageSize = getpagesize();
}

void Machine::addComputeNode(ComputePlace *node){
    _computeNodes[node->getIndex()] = node;
}

void Machine::addMemoryNode(MemoryPlace *node){
    _memoryNodes[node->getIndex()] = node;
}

// TODO reconsider preloading saving vector as a member if there are no modification needs
std::vector<int>* Machine::getComputeNodeIndexes(){
	std::vector<int>* indexes = new std::vector<int>();

	for(ComputeNodes_t::iterator it = _computeNodes.begin(); it != _computeNodes.end(); ++it){
		indexes->push_back(it->first);
	}

	return indexes;
}

std::vector<int>* Machine::getMemoryNodeIndexes(){
	std::vector<int>* indexes = new std::vector<int>();

	for(MemoryNodes_t::iterator it = _memoryNodes.begin(); it != _memoryNodes.end(); ++it){
		indexes->push_back(it->first);
	}

	return indexes;
}

std::vector<ComputePlace*>* Machine::getComputeNodes(){
	std::vector<ComputePlace*>* nodes = new std::vector<ComputePlace*>();

	for(ComputeNodes_t::iterator it = _computeNodes.begin(); it != _computeNodes.end(); ++it){
		nodes->push_back(it->second);
	}

	return nodes;
}

std::vector<MemoryPlace*>* Machine::getMemoryNodes(){
	std::vector<MemoryPlace*>* nodes = new std::vector<MemoryPlace*>();

	for(MemoryNodes_t::iterator it = _memoryNodes.begin(); it != _memoryNodes.end(); ++it){
		nodes->push_back(it->second);
	}

	return nodes;
}

#endif //MACHINE_CPP
