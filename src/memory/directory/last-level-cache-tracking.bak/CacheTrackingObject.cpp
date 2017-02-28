#include "CacheTrackingObject.hpp"

CacheTrackingObject::CacheTrackingObject(DataAccessRange range)
	: _range(range), _lastUse(0) 
{

}

CacheTrackingObject::CacheTrackingObject(const CacheTrackingObject &obj){
	_range = DataAccessRange( obj._range.getStartAddress(), obj._range.getEndAddress() );
}

DataAccessRange &CacheTrackingObject::getAccessRange(){
	return _range;
}

DataAccessRange const &CacheTrackingObject::getAccessRange() const{
	return _range;
}

void *CacheTrackingObject::getStartAddress(){
	return _range.getStartAddress();
}

void *CacheTrackingObject::getEndAddress(){
	return _range.getEndAddress();
}

void CacheTrackingObject::setStartAddress(void *startAddress){
	_range = DataAccessRange(startAddress, _range.getEndAddress());
}

void CacheTrackingObject::setEndAddress(void *endAddress){
	_range = DataAccessRange(_range.getStartAddress(), endAddress);
}

size_t CacheTrackingObject::getSize(){
	return _range.getSize();
}

long unsigned int CacheTrackingObject::getLastUse(){
    return _lastUse;
}

void CacheTrackingObject::updateLastUse(long unsigned int value) {
    _lastUse = value;
}
