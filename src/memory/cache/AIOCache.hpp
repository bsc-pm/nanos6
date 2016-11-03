#ifndef AIO_CACHE_HPP
#define AIO_CACHE_HPP

#include "GenericCache.hpp" 

class AIOCache: public GenericCache {
public:
    AIOCache() {}
    virtual ~AIOCache() {}
    virtual void * allocate(std::size_t size);
    virtual void deallocate(void * ptr);
    virtual void copyData(unsigned int sourceCache, unsigned int homeNode, Task * task);
    virtual void flush(); 
    virtual bool evict();
};

#endif //AIO_CACHE_HPP
