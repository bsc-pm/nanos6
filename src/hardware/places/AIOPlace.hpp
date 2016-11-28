#ifndef AIO_PLACE_HPP
#define AIO_PLACE_HPP

#include "MemoryPlace.hpp"
#include "memory/cache/AIOCache.hpp"

class AIOPlace: public MemoryPlace {
public:
	AIOPlace(int index, AIOCache * cache, AddressSpace * addressSpace = nullptr)
        : MemoryPlace(index, cache, addressSpace)
    {}
    
    virtual ~AIOPlace() {}
    virtual inline AIOCache * getCache() { return (AIOCache *) _cache; }
};

#endif //AIO_PLACE_HPP
