#include "CopyObject.hpp"

CopyObject::CopyObject(DataAccessRange range, int homeNode, int version)
	: _version(version), 
	_caches(),
    _homeNode(homeNode),
    _homeNodeUpToDate(false),
    _range(range)
{

}

CopyObject::CopyObject(const CopyObject &obj){
	_range = DataAccessRange( obj._range.getStartAddress(), obj._range.getEndAddress() );
	_version = obj._version;
	_caches = obj._caches;
    _homeNode = obj._homeNode;
    _homeNodeUpToDate = obj._homeNodeUpToDate;
    assert(!_caches.test(_homeNode) && "homeNode cache cannot be true");
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
    _homeNodeUpToDate = false;
	_version++;
}

void CopyObject::addCache(int id){
    assert(id != _homeNode && "bit concerning homeNode cache cannot be 1");
	_caches.set(id);
}

void CopyObject::removeCache(int id){
	_caches.reset(id);
    assert((anyCache() || _homeNodeUpToDate) && "Data must be at least in one cache or in the homeNode");
}	
 
bool CopyObject::testCache(int id){
	return _caches.test(id);	
}

bool CopyObject::isOnlyCache(int id){
    return _caches.to_ulong() == (unsigned long) (1<<id);
	//if(!_caches.test(id)) return false;

	//for(int i = 0; i < _caches.size(); i++){
	//	if(i != id && _caches.test(id)){
	//		return false;
	//	}
	//}

	//return true;
}

bool CopyObject::anyCache(){
	return _caches.any();
}

int CopyObject::countCaches(){
	return _caches.size();
}	

cache_mask CopyObject::getCaches() {
    return _caches;
}

int CopyObject::getHomeNode() {
    return _homeNode;
}

void CopyObject::setHomeNodeUpToDate(bool b) {
    _homeNodeUpToDate = b;
}

bool CopyObject::isHomeNodeUpToDate() {
    return _homeNodeUpToDate;
}
