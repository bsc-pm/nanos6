#include "MemoryPageObject.hpp"

MemoryPageObject::MemoryPageObject( void *baseAddress, size_t size, MemoryPlace *location = nullptr ): _range(startAddress, size), _location( location ){}

void *MemoryPageObject::getBaseAddress(){
	return _region.getStartAddress();
}

size_t MemoryPageObject::getSize(){
	return _range.getSize();
}

MemoryPlace *MemoryPageObject::getLocation(){
	return location;
}

