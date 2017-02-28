#include "CacheTrackingObject.hpp"

CacheTrackingObject::CacheTrackingObject(DataAccessRange range)
	: _key(range, 0) 
{

}

CacheTrackingObject::CacheTrackingObject(CacheTrackingObjectKey key)
    : _key(key)
{
}

CacheTrackingObject::CacheTrackingObject(const CacheTrackingObject &obj){
	_key._range = DataAccessRange( obj._key._range.getStartAddress(), obj._key._range.getEndAddress() );
    _key._lastUse = obj._key._lastUse;
}

DataAccessRange &CacheTrackingObject::getAccessRange(){
	return _key._range;
}

DataAccessRange const &CacheTrackingObject::getAccessRange() const{
	return _key._range;
}

void *CacheTrackingObject::getStartAddress(){
	return _key._range.getStartAddress();
}

void *CacheTrackingObject::getEndAddress(){
	return _key._range.getEndAddress();
}

void CacheTrackingObject::setStartAddress(void *startAddress){
	_key._range = DataAccessRange(startAddress, _key._range.getEndAddress());
}

void CacheTrackingObject::setEndAddress(void *endAddress){
	_key._range = DataAccessRange(_key._range.getStartAddress(), endAddress);
}

size_t CacheTrackingObject::getSize(){
	return _key._range.getSize();
}

long unsigned int CacheTrackingObject::getLastUse(){
    return _key._lastUse;
}

void CacheTrackingObject::updateLastUse(long unsigned int value) {
    _key._lastUse = value;
}
