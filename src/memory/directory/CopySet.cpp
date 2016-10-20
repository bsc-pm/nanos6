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
	CopySet::iterator it = _set.find(address);
	while(it != _set.end() && it->getStartAddress() == address){
		if(it->getSize() == size){
			return it;
		}
		it++;
	}
	return _set.end();	
}

CopySet::iterator CopySet::insert(void *address, size_t size, int cache, bool increment){
	CopySet::iterator it = find(address, size);
	if(it != _set.end()){
		if(!it->testCache(cache)) it->addCache(cache);
		if(increment) it->incrementVersion();
		return it;
	}	

	CopyObject *cpy = new CopyObject(address, size);
	cpy->addCache(cache);	
	it = _set.insert(*cpy);
	return it;
}

CopySet::iterator CopySet::erase(void *address, size_t size, int cache){
	CopySet::iterator it = _set.find(address);
	if(it != _set.end()){
		it->removeCache(cache);
		if(!it->anyCache()) _set.erase(it);
	}
	return it;	
}
