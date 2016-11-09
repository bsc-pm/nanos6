#include "CopyObject.hpp"

CopyObject::CopyObject(DataAccessRange range, int version)
	: _range(range), 
	_version(version), 
	_caches()
{

}

CopyObject::CopyObject(const CopyObject &obj){
	_range = DataAccessRange( obj._range.getStartAddress(), obj._range.getEndAddress() );
	_version = obj._version;
	_caches = obj._caches;
}

DataAccessRange &CopyObject::getAccessRange(){
	return _range;
}

DataAccessRange const &CopyObject::getAccessRange() const{
	return _range;
}

void *CopyObject::getStartAddress(){
	return _range.getStartAddress();
}

void *CopyObject::getEndAddress(){
	return _range.getEndAddress();
}

void CopyObject::setStartAddress(void *startAddress){
	_range = DataAccessRange(startAddress, _range.getEndAddress());
}

void CopyObject::setEndAddress(void *endAddress){
	_range = DataAccessRange(_range.getStartAddress(), endAddress);
}

size_t CopyObject::getSize(){
	return _range.getSize();
}

int CopyObject::getVersion(){
	return _version;
}

void CopyObject::incrementVersion(){
	_caches.reset();
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

bool CopyObject::isOnlyCache(int id){
	if(!_caches.test(id)) return false;

	for(int i = 0; i < _caches.size(); i++){
		if(i != id && _caches.test(id)){
			return false;
		}
	}

	return true;
}

bool CopyObject::anyCache(){
	return _caches.any();
}

int CopyObject::countCaches(){
	return _caches.size();
}	
