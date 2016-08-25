#ifndef MACHINE_HPP
#define MACHINE_HPP

#include <vector>
#include <map>

#include "Loader.hpp"
#include "places/MemoryPlace.hpp"
#include "places/ComputePlace.hpp"

class Machine {

friend class Loader; // only the loader should modify the machine at startup


private:

	typedef float** distances_t;	
	typedef std::map<int, MemoryPlace*> nodes_t;
	
	size_t _nNodes;
	nodes_t _nodes;
	distances_t _distances;

	long _pagesize; //< pre loaded page size to avoid redundant system calls

	static Machine *_instance;

	Machine()
		:_nNodes(0)
	{
	}

	void _addNode(MemoryPlace *node){
		_nodes[node->_index] = node;
		_nNodes++;
	}
public:
	static Machine *__attribute__ ((noinline)) instance();
	size_t getNodeCount(void);
	MemoryPlace* getNode(int os_index);
        std::vector<int>* getNodeIndexes();
	std::vector<MemoryPlace*>* getNodes();
	nodes_t::iterator nodesBegin(void);
	nodes_t::iterator nodesEnd(void);
	const long getPageSize(void);
};

#endif //MACHINE_HPP
