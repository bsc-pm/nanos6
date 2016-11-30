#ifndef AIO_CACHE_CPP
#define AIO_CACHE_CPP

#include "AIOCache.hpp" 

void * AIOCache::allocate(std::size_t size) {
}

void AIOCache::deallocate(void * ptr) {
}

void AIOCache::copyData(float * cachesLoad, Task * task, unsigned int copiesToDo) {
}

void AIOCache::flush() {
}

bool AIOCache::evict() {
}

void AIOCache::writeBack(void * address) {
}

#endif //AIO_CACHE_CPP
