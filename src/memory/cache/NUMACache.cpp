#ifndef NUMA_CACHE_CPP
#define NUMA_CACHE_CPP

#include "NUMACache.hpp" 

void * NUMACache::allocate(std::size_t size) {
    return  malloc(size);
}

void NUMACache::free(void * ptr) {
    free(ptr);
}

void NUMACache::copyIn(void * devAddress, void * hostAddress, std::size_t size, TransferOps ops) {
    // CopyIn
    replicaInfo_t new_copy;
    new_copy._physicalAddress = devAddress;
    // Increment current version
    new_copy._version = -1; // directory.getVersion(hostAddress)+1
    _replicas[hostAddress] = new_copy; 
    memcpy( devAddress, hostAddress, size );
    ops.completeOp();
    // Increment refCount?
}

void NUMACache::copyOut(void * hostAddress, void * devAddress, std::size_t size, TransferOps ops) {
    // CopyOut
    memcpy( hostAddress, devAddress, size );
    ops.completeOp();
    // Decrement refCount?
}

void NUMACache::copyDev2Dev(void * devDstAddress, void * devSrcAddress, std::size_t size, TransferOps ops) {
    // CopyDev2Dev
    // Probably, I should know if it is in or out
}

#endif //NUMA_CACHE_CPP
