#ifndef ADDRESS_REGION_HPP
#define ADDRESS_REGION_HPP

#include <cstddef> //size_t

struct Region {
	void *_baseAddress;
	void *_endAddress;
	size_t _size;

 	Region(void *baseAddress, size_t size);	

	static inline void *add(void *address, size_t size);
	static inline void *sub(void *address, size_t size);
};

typedef struct Region Region;

#endif //ADDRESS_REGION_HPP
