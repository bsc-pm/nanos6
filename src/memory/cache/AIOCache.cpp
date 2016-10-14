#ifndef AIO_CACHE_CPP
#define AIO_CACHE_CPP

#include "AIOCache.hpp" 

void * AIOCache::allocate(std::size_t size) {
}

void AIOCache::free(void * ptr) {
}

void AIOCache::copyIn(void * devAddress, void * hostAddress, std::size_t size, TransferOps ops) {
    // CopyIn
}
void AIOCache::copyOut(void * hostAddress, void * devAddress, std::size_t size, TransferOps ops) {
    // CopyOut
}
void AIOCache::copyDev2Dev(void * devDstAddress, void * devSrcAddress, std::size_t size, TransferOps ops) {
    // CopyDev2Dev
}

#endif //AIO_CACHE_CPP
