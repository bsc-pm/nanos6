#include "CopySet.hpp"

CopySet::iterator CopySet::begin(){
	return _set.begin();
}

CopySet::iterator CopySet::end(){
	return _set.end();
}

CopySet::iterator CopySet::find(void *address){
	return _set.find(address);
}

CopySet::iterator insert(void *baseAddress, size_t size, GenericCache *cache, bool increment){
	CopySet::iterator it = _set.find(baseAddress);
	if(it != _set.end()){
		//Insert and return
		while(it->getBaseAddress() == baseAddress){
			if(it->getSize() == size){ //same region
				if(!it->isInCache(cache)){
					it->addCache(cache);
				}
				if(increment){
					it->incrementVersion();
				}
				return it;
			}	
		}
	}
	
	std::pair<CopySet::iterator, bool> result = _set.insert(CopyObject(baseAddress, size));
	_set.addCache(cache);
	return result->first;
}

CopySet::iterator erase(void *baseAddress, size_t size, GenericCache *cache){

}
