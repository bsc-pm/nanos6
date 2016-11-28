#ifndef NUMA_CACHE_HPP
#define NUMA_CACHE_HPP

#include "GenericCache.hpp" 

class NUMACache: public GenericCache {
public:
    NUMACache(int index)
        : GenericCache(index) 
    {
    }
    ~NUMACache(){
        //Iterate over _replicas and free all the replicas.
        for(const auto& replica : _replicas ) {
            deallocate(replica.second._physicalAddress);
        }
    }
    virtual void * allocate(std::size_t size);
    virtual void deallocate(void * ptr);
    virtual void copyData(int sourceCache, Task * task, unsigned int copiesToDo);
    virtual void flush(); 
    virtual bool evict();
    virtual void verboseMsg(std::string msg) {
        std::cout << "##### [CACHE " << _index << "]: " << msg << " . #####" << std::endl;
    }
    virtual void writeBack(void*);
};

#endif //NUMA_CACHE_HPP
