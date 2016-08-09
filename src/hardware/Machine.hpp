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
			
		//Call loader	

		_pagesize = getpagesize();
	}

	static const MemoryPlace* getNode(int os_index){
		return _nodes[os_index];
	}

	static const ComputePlace** getCpusByNode(int os_index){
		return _nodes[os_index]->_cpus;
	}

	static const long getpagesize(void){
		return _pagesize;
	}

        // return a vector with the os_index of the nodes
        static const vector<int>* getNodeIndexes(){

                std::vector<int> indexes = new std::vector<int>();

                for(nodes_t::iterator it = _nodes.begin(); it != _nodes.end(); ++it){
                        v.push_back(it->first);
                }

                return indexes;
        }

};

#endif //MACHINE_HPP
