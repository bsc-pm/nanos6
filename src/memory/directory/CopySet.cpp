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
	while(it != _set.end() && it->getStartAddress() > address){
		if(it->getSize() == size){
			return it;
		}
		it++;
	}
	return _set.end();	
}

CopySet::iterator CopySet::insert(void *address, size_t size, int cache, bool increment){
/*	CopySet::iterator it = _set.lower_bound(address);
    CopySet::iterator initial = it;
	if(it != set.end()){
		//Adjust lower bound
		if ((it != _set.begin()) && (it->getStartAddress() > address)) {
            it--;
        }
		
		//If previous element is not intersecting, readjust
		if( it->getEndAddress() <= address ){
			it = initial;
		}
		
		void *lastEnd = address;
		void *endAddress;
		while( it != _set.end() && endAddress > it->getStartAddress()){
			CopySet::iterator position = it;
			it++;
			
			if(lastEnd < position->getStartAddress()){
				//Misssing region before a position
				CopyObject *cpy = new CopyObject(address, size);
				//DataAccessRange missingRange(lastEnd, position->getStartAddress());				
			}
			
			if(position->getEndAddress() <= endAddress){
				//Intersecting region
				lastEnd = position->getEndAddress();			

			} else {
				//Intersecting region
				lastEnd = endAddress();
			} 
			
		}

		if(lastEnd < endAddress){
			//Missing region at the end
		}

    } else {
		CopyObject *cpy = new CopyObject(address, size);
		cpy->addCache(cache);	
		it = _set.insert(*cpy);
		return it;
	}

	

	

	CopySet::iterator it = find(address, size);
	if(it != _set.end()){
		if(!it->testCache(cache)) it->addCache(cache);
		if(increment) it->incrementVersion();
		return it;
	}
*/	
}



CopySet::iterator CopySet::erase(void *address, int cache){
	CopySet::iterator it = _set.find(address);
	if(it != _set.end()){
		it->removeCache(cache);
		if(!it->anyCache()) _set.erase(it);
	}
	return it;	
}
