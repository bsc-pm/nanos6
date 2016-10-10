#include "Region.hpp"

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
