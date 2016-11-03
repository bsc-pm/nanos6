#include "CopySet.hpp"

CopySet::CopySet(): _set(){}

/* Private methods */

void CopySet::processMissing(void *startAddress, void *endAddress, int cache, bool increment){
	size_t size = static_cast<char *>(endAddress) - static_cast<char *>(startAddress);
	CopyObject *cpy = new CopyObject(startAddress, size);
	cpy->addCache(cache);
	if(increment) cpy->incrementVersion();
	_set.insert(*cpy);
}

void CopySet::processIntersecting(CopyObject &cpy, void *startAddress, void *endAddress, int cache, bool increment){
	if(cpy.getStartAddress() < startAddress){
		CopyObject cpy = new CopyObject(cpy->getStartAddress(), startAddress);
		cpy.addCache(cache);
		_set.insert(*cpy);

		cpy->setStartAddress(startAddress); 
	}

	if(cpy->getEndAddress() > endAddress){
		CopyObject *cpy = new CopyObject(cpy->getEndAddress(), endAddress);
		cpy.addCache(cache);
		_set.insert(*cpy);

		cpy->setEndAddress(endAddress);
	}

	if(increment) cpy->incrementVersion();
	cpy->addCache(cache);

}

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
	while(it != _set.end() && it->getStartAddress() > address){
		if(it->getSize() == size){
			return it;
		}
		it++;
	}
	return _set.end();	
}

CopySet::iterator CopySet::insert(void *startAddress, size_t size, int cache, bool increment){
	void *endAddress = static_cast<void *>( static_cast<char *>(startAddress) + size )
	CopySet::iterator it = _set.lower_bound(startAddress);
    CopySet::iterator initial = it;

	if(it != set.end()){
		//Adjust lower bound
		if ((it != _set.begin()) && (it->getStartAddress() > address)) {
            it--;
        }
		
		//If previous element is not intersecting, readjust
		if( it->getEndAddress() <= startAddress ){
			it = initial;
		}
		
		void *lastEnd = startAddress;
		while( it != _set.end() && endAddress > it->getStartAddress()){
			CopySet::iterator position = it;
			it++;
			
			if(lastEnd < position->getStartAddress()){
				//Misssing region before a position
				processMissing(lastEnd, position->getStartAddress);				
			}
			
			if(position->getEndAddress() <= endAddress){
				//Intersecting region
				lastEnd = position->getEndAddress();			
				processIntersecting(*position, startAddress, endAddress, cache, increment);
			} else {
				//Intersecting region
				lastEnd = endAddress;
				processIntersecting(*position, startAddress, endAddress, cache, increment);
			} 
			
		}

		if(lastEnd < endAddress){
			//Missing region at the end
			//If not intersecting this is the whole region
			processMissing(lastEnd, endAddress);
		}

    } else {
		processMissing(startAddress, endAddress);
	}
}


CopySet::iterator CopySet::erase(void *address, int cache){
	CopySet::iterator it = _set.find(address);
	if(it != _set.end()){
		it->removeCache(cache);
		if(!it->anyCache()) _set.erase(it);
	}
	return it;	
}
