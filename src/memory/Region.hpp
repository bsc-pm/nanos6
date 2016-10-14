#ifndef ADDRESS_REGION_HPP
#define ADDRESS_REGION_HPP

#include <cstddef> //size_t

struct Region {
	void *_baseAddress;
	void *_endAddress;
	size_t _size;

 	Region(void *baseAddress, size_t size);	
	void **pages();
	unsigned long pageCount();

	bool intersects(Region &other);
	bool contains(Region &other);
	bool containedIn(Region &other);
	
	static inline void *add(void *address, size_t size);
	static inline void *sub(void *address, size_t size);
	static inline size_t distance(void *ptr1, void *ptr2);

	
};

typedef struct Region Region;

#endif //ADDRESS_REGION_HPP
