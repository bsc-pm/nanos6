#include "CopyObject.hpp"

CopyObject::CopyObject(void *startAddress, size_t length): _range(startAddress, length), version(0), caches(){}

void CopyObject::*getStartAddress(){
	return _range.getStartAddress();
}

size_t CopyObject::getSize(){
	return _range.getSize();
}

int CopyObject::getVersion(){
	return _version;
}

void CopyObject::setVersion(int version){
	_version = version;
}

void CopyObject::incrementVersion(){
	_version++;
}

void CopyObject::addCache(GenericCache *cache){
	_caches.add(cache);
}

void CopyObject::removeCache(GenericCache *cache){
	_caches.erase(_caches.find(cache));
}	
 
bool CopyObject::isInCache(GenericCache *cache){
	return _caches.count(cache) != 0;	
}

bool CopyObject::countCaches(){
	return _caches.size();
}	
