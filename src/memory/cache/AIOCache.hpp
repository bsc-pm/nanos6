#ifndef AIO_CACHE_HPP
#define AIO_CACHE_HPP

#include "GenericCache.hpp" 

class AIOCache: public GenericCache {
public:
    AIOCache(int index)
        : GenericCache(index) 
    {}
    virtual ~AIOCache() {}
    virtual void * allocate(std::size_t size);
    virtual void deallocate(void * ptr);
    virtual void copyData(float * cachesLoad, Task * task, unsigned int copiesToDo);
    virtual void flush(); 
    virtual bool evict();
    virtual void writeBack(void * address);
};

#endif //AIO_CACHE_HPP
