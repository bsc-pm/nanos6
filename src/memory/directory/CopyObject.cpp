#include "CopyObject.hpp"

CopyObject::CopyObject(void *startAddress, size_t length): _range(startAddress, length), _version(0), _caches(){}

void *CopyObject::getStartAddress(){
	return _range.getStartAddress();
}

size_t CopyObject::getSize(){
	return _range.getSize();
}

int CopyObject::getVersion(){
	return _version;
}

void CopyObject::incrementVersion(){
	_version++;
}

void CopyObject::addCache(GenericCache *cache){
	_caches.insert(cache);
}

void CopyObject::removeCache(GenericCache *cache){
	_caches.erase(_caches.find(cache));
}	
 
bool CopyObject::isInCache(GenericCache *cache){
	return _caches.count(cache) != 0;	
}

int CopyObject::countCaches(){
	return _caches.size();
}	
