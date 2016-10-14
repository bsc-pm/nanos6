#ifndef GENERIC_CACHE_HPP
#define GENERIC_CACHE_HPP

#include <map>
#include <string.h>
//#include <boost/pool/pool_alloc.hpp>
#include "../support/TransferOps.hpp"
//#include "CacheObject.hpp"

class GenericCache {
public: 
    struct replicaInfo_t {
        void * _physicalAddress;
        int _version;
        //int _refCount;
    };
protected:
    typedef std::map<void *, replicaInfo_t> replicas_t;
    // List of replicas. The key is the logical address in the OmpSs address space (i.e. the address of the user space). The value is the physical address 
    // where the replica is actually stored in the cache and the version of the replica.
    replicas_t _replicas;
    //boost::pool_allocator<char> _pool;
public:
    GenericCache() {}
    virtual ~GenericCache() {}
    virtual void * allocate(std::size_t size) = 0;
    virtual void free(void * ptr) = 0;
    virtual void copyIn(void * devAddress, void * hostAddress, std::size_t size, TransferOps ops) = 0;
    virtual void copyOut(void * hostAddress, void * devAddress, std::size_t size, TransferOps ops) = 0;
    virtual void copyDev2Dev(void * devDstAddress, void * devSrcAddress, std::size_t size, TransferOps ops) = 0;
};

#endif //GENERIC_CACHE_HPP
