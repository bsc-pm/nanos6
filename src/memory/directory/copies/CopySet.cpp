#include "CopySet.hpp"
#include <DataAccessRange.hpp>
#include <iostream>

CopySet::CopySet(): _set(){}

/* Private methods */

CopySet::iterator CopySet::begin(){
	return _set.begin();
}

CopySet::iterator CopySet::end(){
	return _set.end();
}

bool CopySet::empty(){
	return _set.empty();
}

void CopySet::insert(void *startAddress, size_t size, int cache, bool increment){
	
	DataAccessRange range(startAddress, size);


	//First version of insert with defrag. 
	if(!increment){	
		// When CopyIn:
		// Every region intersecting is added the cache.
		// Every missing region is created and added the cache.
		// Regions on the edges are shrunk and the cache is added to them.
		

		// Process Interesection section, shrinking edges and adding a new cache to each old region.
		_set.processIntersecting(
			range,
			[&] (CopySet::iterator position) -> bool {	
				if(position->getStartAddress() < range.getStartAddress()){
					if(position->getEndAddress() > range.getEndAddress()){
						CopyObject *cpy = new CopyObject( *position );
						cpy->setStartAddress(range.getEndAddress() );
						_set.insert(*cpy);
					}
					position->setEndAddress(range.getStartAddress()); 
				}
				if(position->getEndAddress() > range.getEndAddress()){
					position->setStartAddress(range.getEndAddress());
				}	
				if(position->getEndAddress() <= range.getEndAddress() && position->getStartAddress() >= range.getStartAddress()){
					position->addCache(cache);
				}
				return true;
			}
		);
		
		// Process Missing sections, creating a copyObject with the added cache for each.
		_set.processMissing(
			range,
			[&] (DataAccessRange missingRange) -> bool {
				CopyObject *cpy = new CopyObject(missingRange, 0);
				cpy->addCache(cache);
				_set.insert(*cpy);
				return true;
			} 
		);
	} else {
		// When CopyOut (incrementing version):
		// All regions are absorbed into the new one. 
		// Regions on the edges are shrunk (if needed).
		int version = 0;

		// Process all intersecting sections, resizing the edges as needed.
		_set.processIntersecting(
			range,
			[&] (CopySet::iterator position) -> bool {
				if(position->getVersion() > version){
					version > position->getVersion();
				}
				 	
				bool edge = false;	
				if(position->getStartAddress() < range.getStartAddress()){
					if(position->getEndAddress() > range.getEndAddress()){
						CopyObject *cpy = new CopyObject( *position );
						cpy->setStartAddress(range.getEndAddress() );
						_set.insert(*cpy);
					}
					position->setEndAddress(range.getStartAddress());
					edge = true; 
				}
				if(position->getEndAddress() > range.getEndAddress()){
					position->setStartAddress(range.getEndAddress());
					edge = true;
				}
				if(!edge){
					_set.erase(*position);
					delete &(*position);
				}

				return true;
			}
		);
			
		CopyObject *cpy = new CopyObject( range, version + 1 );
		cpy->addCache(cache);
		_set.insert(*cpy);
	}
}


CopySet::iterator CopySet::erase(void *address, int cache){

 	DataAccessRange eraseRange(address, address);	
	CopySet::iterator it = _set.find(eraseRange);
	it->removeCache(cache);
	if(!it->anyCache()){
		_set.erase(*it);
	}
	
	return it;	
}
