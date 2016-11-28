#ifndef NUMA_PLACE_HPP
#define NUMA_PLACE_HPP

#include "MemoryPlace.hpp"
#include "memory/cache/NUMACache.hpp"

class NUMAPlace: public MemoryPlace {
private:
	typedef std::map<int, ComputePlace*> processingUnits_t;
	processingUnits_t _processingUnits; //ProcessingUnits able to interact with this MemoryPlace
public:
	NUMAPlace(int index, NUMACache * cache, AddressSpace * addressSpace = nullptr)
        : MemoryPlace(index, cache, addressSpace)
	{}
    
    virtual ~NUMAPlace() {}
    virtual inline NUMACache * getCache() { return (NUMACache *) _cache; }
	size_t getPUCount(void) const { return _processingUnits.size(); }
	ComputePlace* getPU(int index){ return _processingUnits[index]; }
	void addPU(ComputePlace* pu);
	std::vector<int> getPUIndexes();
	std::vector<ComputePlace*> getPUs();
};

#endif //NUMA_PLACE_HPP
