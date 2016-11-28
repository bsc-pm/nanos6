#ifndef MACHINE_HPP
#define MACHINE_HPP

#include <vector>
#include <map>
#include <atomic>

#include "places/MemoryPlace.hpp"
#include "places/ComputePlace.hpp"

class Machine {
public: 
	typedef std::map<int, MemoryPlace*> MemoryNodes_t;
	typedef std::map<int, ComputePlace*> ComputeNodes_t;
private:
    // pair.first -> ComputePlace || pair.second -> MemoryPlace
    typedef std::map<std::pair<int, int>, float> Distances_t;
	
    // Physical memory places. For instance, 2 NUMA nodes are 2 different memory places.
    MemoryNodes_t _memoryNodes;
    // Physical compute places.
    ComputeNodes_t _computeNodes;
    // Cost of accessing from a ComputePlace to a MemoryPlace.
    Distances_t _distances;

	long _pageSize; //< pre loaded page size to avoid redundant system calls

    // The actual instance of this class.
	static Machine *_machine;

	Machine()
	{
	}

public:
    // Generic methods
    static void initialize();
	static inline Machine * getMachine() { return _machine; }
	static inline long getPageSize(void) { return _machine->_pageSize; }

    // ComputeNodes related methods
	static inline size_t getComputeNodeCount(void) { return _machine->_computeNodes.size(); }
	static void addComputeNode(ComputePlace *node);
	static inline ComputePlace* getComputeNode(int index) { return _machine->_computeNodes[index]; }
    static std::vector<int> getComputeNodeIndexes();
	static std::vector<ComputePlace*> getComputeNodes();
	static inline ComputeNodes_t::iterator computeNodesBegin(void) { return _machine->_computeNodes.begin(); }
	static inline ComputeNodes_t::iterator computeNodesEnd(void) { return _machine->_computeNodes.end(); }

    // MemoryNodes related methods
	static inline size_t getMemoryNodeCount(void) { return _machine->_memoryNodes.size(); }
	static void addMemoryNode(MemoryPlace *node);
	static inline MemoryPlace* getMemoryNode(int index) { return _machine->_memoryNodes[index]; }
    static std::vector<int> getMemoryNodeIndexes();
	static std::vector<MemoryPlace*> getMemoryNodes();
	static inline MemoryNodes_t::iterator memoryNodesBegin(void) { return _machine->_memoryNodes.begin(); }
	static inline MemoryNodes_t::iterator memoryNodesEnd(void) { return _machine->_memoryNodes.end(); }
};

#endif //MACHINE_HPP
