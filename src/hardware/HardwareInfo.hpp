#ifndef HARDWARE_INFO_HPP
#define HARDWARE_INFO_HPP
#include <vector>
#include <map>

#include "places/MemoryPlace.hpp"
#include "places/ComputePlace.hpp"

class HardwareInfo {
private:
	typedef std::vector<MemoryPlace*> memory_nodes_t;
	// When new compute places are added, this can become a vector of vectors ([device type][device id])
	typedef std::vector<ComputePlace*> compute_nodes_t;
	
	// \brief Physical memory places. For instance, 2 NUMA nodes are 2 different memory places.
	static memory_nodes_t _memoryNodes;
	// \brief Logical compute places.
	static compute_nodes_t _computeNodes;

public:
	// Generic methods
	static void initialize();
	static void shutdown();

	// ComputeNodes related methods
	// When new compute places are added, a new parameter type could be added
	static inline size_t getComputeNodeCount(void)
	{
		return _computeNodes.size();
	}

	static inline ComputePlace* getComputeNode(int index)
	{
		return _computeNodes[index];
	}

	static inline std::vector<ComputePlace*> const &getComputeNodes()
	{
		return _computeNodes;
	}

	// MemoryNodes related methods
	static inline size_t getMemoryNodeCount(void)
	{
		return _memoryNodes.size();
	}

	static inline MemoryPlace* getMemoryNode(int index)
	{
		return _memoryNodes[index];
	}

	static inline std::vector<MemoryPlace*> const &getMemoryNodes()
	{
		return _memoryNodes;
	}
};

#endif // HARDWARE_INFO_HPP
