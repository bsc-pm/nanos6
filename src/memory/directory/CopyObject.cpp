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

void CopyObject::addCache(int id){
	_caches.set(id);
}

void CopyObject::removeCache(int id){
	_caches.reset(id);
}	
 
bool CopyObject::testCache(int id){
	return _caches.test(id);	
}

bool CopyObject::anyCache(){
	return _caches.any();
}

int CopyObject::countCaches(){
	return _caches.size();
}	
