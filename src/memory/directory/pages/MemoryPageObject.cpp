#include "MemoryPageObject.hpp"

MemoryPageObject::MemoryPageObject( void *startAddress, size_t size, int location ): _range(startAddress, size), _location( location ){}

void *MemoryPageObject::getStartAddress(){
	return _range.getStartAddress();
}

size_t MemoryPageObject::getSize(){
	return _range.getSize();
}

int MemoryPageObject::getLocation(){
	return _location;
}

