#include "Region.hpp"

unsigned int Region::pageCount(){
	return _size / Machine::getMachine()->getPageSize();
}

void Region::pages(void **pages){
	long pagesize = Machine::getMachine()->getPageSize();
	count = pageCount();	

	pages[0] = (void *)( (long) _baseAddress & ~(pagesize-1) );
	for(int i = 0; i < count; i++){
		pages[i] = pages[0] + i * pagesize;
	}

}


Region::Region(void *baseAddress, size_t size)
	: _baseAddress(baseAddress),
	_endAddress(Region::add(baseAddress, size)),
	_size(size){
		
}

void *Region::add(void *ptr, size_t bytes){
	return static_cast<void *>( static_cast<char *>( ptr ) + bytes );
}

void *Region::sub(void *ptr, size_t bytes){
	return static_cast<void *>( static_cast<char *>( ptr ) - bytes );
}

size_t *Region::distance(void *ptr1, void *ptr2){
	return static_cast<char *>(ptr1) - static_cast<char *>(ptr2);
}
