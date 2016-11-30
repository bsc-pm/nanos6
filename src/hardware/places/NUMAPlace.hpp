#ifndef NUMA_PLACE_HPP
#define NUMA_PLACE_HPP

#include "MemoryPlace.hpp"
#include "memory/cache/NUMACache.hpp"

class NUMAPlace: public MemoryPlace {
private:
	typedef std::map<int, ComputePlace*> computePlaces_t;
	computePlaces_t _computePlaces; //ComputePlaces able to interact with this MemoryPlace
public:
	NUMAPlace(int index, NUMACache * cache, AddressSpace * addressSpace = nullptr)
        : MemoryPlace(index, cache, addressSpace)
	{}
    
    virtual ~NUMAPlace() {}
    virtual inline NUMACache * getCache() { return (NUMACache *) _cache; }
	size_t getComputePlaceCount(void) const { return _computePlaces.size(); }
	ComputePlace* getComputePlace(int index){ return _computePlaces[index]; }
	void addComputePlace(ComputePlace* computePlace);
	std::vector<int> getComputePlaceIndexes();
	std::vector<ComputePlace*> getComputePlaces();
};

#endif //NUMA_PLACE_HPP
