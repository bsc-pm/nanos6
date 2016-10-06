#ifndef NUMA_CACHE_HPP
#define NUMA_CACHE_HPP

#include "GenericCache.hpp" 

class NUMACache: public GenericCache {
public:
    NUMACache() {}
    virtual ~NUMACache() {}
    virtual void * allocate(std::size_t size);
    virtual void free(void * ptr);
    virtual void copyIn(void * devAddress, void * hostAddress, std::size_t size, TransferOps ops);
    virtual void copyOut(void * hostAddress, void * devAddress, std::size_t size, TransferOps ops);
    virtual void copyDev2Dev(void * devDstAddress, void * devSrcAddress, std::size_t size, TransferOps ops);
};

#endif //NUMA_CACHE_HPP
