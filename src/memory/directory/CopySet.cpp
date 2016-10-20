#include "CopySet.hpp"

CopySet::CopySet(): _set(){}

CopySet::iterator CopySet::begin(){
	return _set.begin();
}

CopySet::iterator CopySet::end(){
	return _set.end();
}

CopySet::iterator CopySet::find(void *address){
	return _set.find(address);
}

CopySet::iterator CopySet::find(void *address, size_t size){
	CopySet::iterator it = _set.find(baseAddress);
	while(it != _set.end() && it->getStartAddress() == address){
		if(it->getSize() == size){
			return it;
		}
		it++;
	}
	return _set.end();	
}

CopySet::iterator insert(void *baseAddress, size_t size, GenericCache *cache, bool increment){
	CopySet::iterator it = find(baseAddress, size);
	if(it != _set.end()){
		if(!it->isInCache(cache)) it->addCache(cache);
		if(increment) it->incrementVersion();
		return it;
	}	
	
	std::pair<CopySet::iterator, bool> result = _set.insert(CopyObject(baseAddress, size));
	_set.addCache(cache);
	return result->first;
}

CopySet::iterator erase(void *baseAddress, size_t size, GenericCache *cache){
	CopySet::iterator it = _set.find(baseAddress);
	if(it != _set.end()){
		it->removeCache(cache);
		if(it->countCaches() == 0) _set.erase(it);
	}
	return it;	
}
