#ifndef NUMA_PLACE_HPP
#define NUMA_PLACE_HPP

#include "MemoryPlace.hpp"
#include "../../memory/NUMACache.hpp"

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
	const size_t getPUCount(void){ return _processingUnits.size(); }
	const ComputePlace* getPU(int index){ return _processingUnits[index]; }
	void addPU(ComputePlace* pu);
	const std::vector<int>* getPUIndexes();
	const std::vector<ComputePlace*>* getPUs();
};

#endif //NUMA_PLACE_HPP
