#ifndef MACHINE_HPP
#define MACHINE_HPP

#include "loaders/Loader.hpp"
#include "loaders/HWLOCLoader.cpp"

class Machine {

friend class Loader; // only the loader should modify the machine at startup

private:
	typedef std::map<int, MemoryPlace*> nodes_t;
	typedef float** distances_t;	

	size_t _nNodes;
	nodes_t _nodes;
	distances_t _distances;

	static Machine *_instance;

	Machine()
		:_nNodes(0)
	{
	}

	void _addNode(MemoryPlace *node){
		_nodes[os_index] = node;
		_nNodes++;
	}
public:
	//TODO Need to define how data is accesses by the scheduler, etc. to make functions
	static Machine *create(){
		_instance = new Machine();
			
		//Call loader	
	}
};

#endif //MACHINE_HPP
