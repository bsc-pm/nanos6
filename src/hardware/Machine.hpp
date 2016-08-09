#ifndef MACHINE_HPP
#define MACHINE_HPP

#include "loaders/Loader.hpp"
#include "loaders/HWLOCLoader.cpp"

class Machine {

friend class Loader; // only the loader should modify the machine at startup

typedef float** distances_t;	
typedef std::map<int, MemoryPlace*> nodes_t;

private:

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
		_nodes[os_index] = node;
		_nNodes++;
	}
public:

	//TODO Need to define how data is accesses by the scheduler, etc. to make functions
	static Machine *create(){
		_instance = new Machine();
			
		HWLOCLoader loader = new HWLOCLoader();
		loader.load();	

		_pagesize = getpagesize();
	}
	
	static const size_t getNodeCount(void){
		return _nodes.size();
	}

	static const MemoryPlace* getNode(int os_index){
		return _nodes[os_index];
	}

	// TODO reconsider preloading vector if there are no modification needs
        static const vector<int>* getNodeIndexes(){

                std::vector<int>* indexes = new std::vector<int>();

                for(nodes_t::iterator it = _nodes.begin(); it != _nodes.end(); ++it){
                        indexes.push_back(it->first);
                }

                return indexes;
        }

	static const vector<MemoryPlace*>* getNodes(){
		
		std::vector<int>* nodes = new std::vector<int>();
		
		for(nodes_t::iterator it = _nodes.begin(); it != _nodes.end(); ++it){
			nodes.push_back(it->first);
		}

		return nodes;
	}

	static nodes_t::iterator nodesBegin(void){
		return _nodes.begin();
	}

	static nodes_t::iterator nodesEnd(void){
		return _nodes.end();
	}

	static const long getpagesize(void){
		return _pagesize;
	}
};

#endif //MACHINE_HPP
