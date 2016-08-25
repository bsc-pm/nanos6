#include "Machine.hpp"

#include <vector>
#include <map>

#include "Loader.hpp"
#include "places/MemoryPlace.hpp"
#include "places/ComputePlace.hpp"


Machine *Machine::_instance = nullptr;

Machine *Machine::instance(){
	if(instance == nullptr){
		Machine::_instance = new Machine();
		Loader l;
		l.load(Machine::_instance);
		Machine::_instance->_pagesize = getpagesize();
	}

	return Machine::_instance;
}

size_t Machine::getNodeCount(void){
	return _nodes.size();
}

MemoryPlace* Machine::getNode(int os_index){
	return _nodes[os_index];
}

// TODO reconsider preloading saving vector as a member if there are no modification needs
std::vector<int>* Machine::getNodeIndexes(){

	std::vector<int>* indexes = new std::vector<int>();

	for(Machine::nodes_t::iterator it = _nodes.begin(); it != _nodes.end(); ++it){
		indexes->push_back(it->first);
	}

	return indexes;
}

std::vector<MemoryPlace*>* Machine::getNodes(){

	std::vector<MemoryPlace*>* nodes = new std::vector<MemoryPlace*>();

	for(Machine::nodes_t::iterator it = _nodes.begin(); it != _nodes.end(); ++it){
		nodes->push_back(it->second);
	}

	return nodes;
}

Machine::nodes_t::iterator Machine::nodesBegin(void){
	return _nodes.begin();
}

Machine::nodes_t::iterator Machine::nodesEnd(void){
	return _nodes.end();
}

const long Machine::getPageSize(void){
	return _pagesize;
}
