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
    _machine->_computeNodes[node->getIndex()] = node;
}

void Machine::addMemoryNode(MemoryPlace *node){
    _machine->_memoryNodes[node->getIndex()] = node;
}

// TODO reconsider preloading saving vector as a member if there are no modification needs
std::vector<int> Machine::getComputeNodeIndexes(){
	std::vector<int> indexes(_machine->_computeNodes.size());

    int i = 0;
	for(ComputeNodes_t::iterator it = _machine->_computeNodes.begin(); 
        it != _machine->_computeNodes.end(); 
        ++it, ++i)
    {
		//indexes.push_back(it->first);
        indexes[i] = it->first;
	}

	return indexes;
}

std::vector<int> Machine::getMemoryNodeIndexes(){
	std::vector<int> indexes(_machine->_memoryNodes.size());

    int i = 0;
	for(MemoryNodes_t::iterator it = _machine->_memoryNodes.begin(); 
        it != _machine->_memoryNodes.end(); 
        ++it, ++i)
    {
		//indexes.push_back(it->first);
        indexes[i] = it->first;
	}

	return indexes;
}

std::vector<ComputePlace*> Machine::getComputeNodes(){
	std::vector<ComputePlace*> nodes(_machine->_computeNodes.size());

    int i = 0;
	for(ComputeNodes_t::iterator it = _machine->_computeNodes.begin(); 
        it != _machine->_computeNodes.end(); 
        ++it, ++i)
    {
		//nodes.push_back(it->second);
        nodes[i] = it->second;
	}

	return nodes;
}

std::vector<MemoryPlace*> Machine::getMemoryNodes(){
	std::vector<MemoryPlace*> nodes(_machine->_memoryNodes.size());

    int i = 0;
	for(MemoryNodes_t::iterator it = _machine->_memoryNodes.begin(); 
        it != _machine->_memoryNodes.end(); 
        ++it, ++i)
    {
		//nodes.push_back(it->second);
        nodes[i] = it->second;
	}

	return nodes;
}

#endif //MACHINE_CPP
