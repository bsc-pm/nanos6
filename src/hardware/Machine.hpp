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
	inline const long getPageSize(void) { return _pageSize; }

    // ComputeNodes related methods
	inline size_t getComputeNodeCount(void) { return _computeNodes.size(); }
	void addComputeNode(ComputePlace *node);
	inline ComputePlace* getComputeNode(int index) { return _computeNodes[index]; }
    std::vector<int>* getComputeNodeIndexes();
	std::vector<ComputePlace*>* getComputeNodes();
	inline ComputeNodes_t::iterator computeNodesBegin(void) { return _computeNodes.begin(); }
	inline ComputeNodes_t::iterator computeNodesEnd(void) { return _computeNodes.end(); }

    // MemoryNodes related methods
	inline size_t getMemoryNodeCount(void) { return _memoryNodes.size(); }
	void addMemoryNode(MemoryPlace *node);
	inline MemoryPlace* getMemoryNode(int index) { return _memoryNodes[index]; }
    std::vector<int>* getMemoryNodeIndexes();
	std::vector<MemoryPlace*>* getMemoryNodes();
	inline MemoryNodes_t::iterator memoryNodesBegin(void) { return _memoryNodes.begin(); }
	inline MemoryNodes_t::iterator memoryNodesEnd(void) { return _memoryNodes.end(); }
};

#endif //MACHINE_HPP
