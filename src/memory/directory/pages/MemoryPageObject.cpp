#include "MemoryPageObject.hpp"

MemoryPageObject::MemoryPageObject( void *startAddress, size_t size, int location ): _location(location), _range(startAddress, size){}

DataAccessRange &MemoryPageObject::getAccessRange(){
	return _range;
}

DataAccessRange const &MemoryPageObject::getAccessRange() const{
	return _range;
}

void *MemoryPageObject::getStartAddress(){
	return _range.getStartAddress();
}

void MemoryPageObject::setStartAddress(void *address){
	_range = DataAccessRange(address, _range.getEndAddress() );
}

void *MemoryPageObject::getEndAddress(){
	return _range.getEndAddress();
}

void MemoryPageObject::setEndAddress(void *address){
	_range = DataAccessRange(_range.getStartAddress(), address);
}

size_t MemoryPageObject::getSize(){
	return _range.getSize();
}

int MemoryPageObject::getLocation(){
	return _location;
}

