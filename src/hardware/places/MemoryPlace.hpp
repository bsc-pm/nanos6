#ifndef MEMORY_PLACE_HPP
#define MEMORY_PLACE_HPP

#include <vector>
#include <map>
#include "../../memory/AddressSpace.hpp"
#include "../../memory/cache/GenericCache.hpp"

class ComputePlace;

class MemoryPlace {
protected:
    AddressSpace * _addressSpace;
    int _index;	
    GenericCache * _cache;
	
public:
	MemoryPlace(int index, GenericCache * cache, AddressSpace * addressSpace = nullptr)
        : _index(index), _addressSpace(addressSpace), _cache(cache)
	{}
    
    virtual ~MemoryPlace() {}
	inline int getIndex(void){ return _index; } 
    inline AddressSpace * getAddressSpace(){ return _addressSpace; } 
    virtual inline GenericCache * getCache() = 0;
};

#endif //MEMORY_PLACE_HPP
