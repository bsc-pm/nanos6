#ifndef AIO_CACHE_CPP
#define AIO_CACHE_CPP

#include "AIOCache.hpp" 

#define _unused(x) ((void)(x))

void * AIOCache::allocate(std::size_t size) {
    _unused(size);
    return (void *)1;
}

void AIOCache::deallocate(void * ptr) {
    _unused(ptr);
}

void AIOCache::copyData(float * cachesLoad, Task * task, unsigned int copiesToDo) {
    _unused(cachesLoad);
    _unused(task);
    _unused(copiesToDo);
}

void AIOCache::flush() {
}

bool AIOCache::evict() {
    return false;
}

void AIOCache::writeBack(void * address) {
    _unused(address);
}

#endif //AIO_CACHE_CPP
