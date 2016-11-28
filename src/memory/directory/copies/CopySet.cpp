#include "CopySet.hpp"
#include <DataAccessRange.hpp>
#include <iostream>

CopySet::CopySet(): BaseType(){}

CopySet::iterator CopySet::find(void *address){
	return BaseType::find( DataAccessRange(address, address) );
}

int CopySet::insert(void *startAddress, size_t size, int homeNode, int cache, bool increment){
	
	DataAccessRange range(startAddress, size);
	int result = 0;

	// TODO check de-fragmentation cases
	//First version of insert with defrag. 

    CopySet::iterator it = find(startAddress);
    if(it != BaseType::end()) {
        assert(it->getHomeNode() == homeNode && "homeNode must match");
        //! The copy object is already in the directory, just update it.
        //! Do it first because it resets the cache mask.
        if(increment) 
            it->incrementVersion();

        if(cache == homeNode) 
            it->setHomeNodeUpToDate(true);
        else
            it->addCache(cache);

        result = it->getVersion();

        //! If version has been incremented, it can only be in cache or in the homeNode but not in both. 
        assert((increment && (it->isHomeNodeUpToDate() != it->isOnlyCache(cache))) || !increment); 
    }
    else {
        if(!increment){	
            // When CopyIn:
            // Every region intersecting is added the cache.
            // Every missing region is created and added the cache.
            // Regions on the edges are shrunk and the cache is added to them.


            // Process Interesection section, shrinking edges and adding a new cache to each old region.
            BaseType::processIntersecting(
                    range,
                    [&] (CopySet::iterator position) -> bool {	
                        // Check if the current position is a edge. If so shrink. 
                        if(position->getStartAddress() < range.getStartAddress()){
                            // If the leftmost edge is the same as the rightmost (
                            if(position->getEndAddress() > range.getEndAddress()){
                                // TODO merge if already has cache
                                CopyObject *cpy = new CopyObject(*position);
                                cpy->setStartAddress(range.getEndAddress());
                                if(homeNode == cache) {
                                    cpy->setHomeNodeUpToDate(true);
                                    cpy->removeCache(homeNode);
                                }
                                BaseType::insert(*cpy);
                            }
                            position->setEndAddress(range.getStartAddress()); 
                        }
                        if(position->getEndAddress() > range.getEndAddress()){
                            position->setStartAddress(range.getEndAddress());
                        }	
                        if(position->getEndAddress() <= range.getEndAddress() && position->getStartAddress() >= range.getStartAddress()){
                            if(cache != position->getHomeNode())
                                position->addCache(cache);
                            else {
                                position->setHomeNodeUpToDate(true);
                                position->removeCache(homeNode);
                            }

                            //Store the highest versioning copyObject
                            if(result < position->getVersion()){
                                result = position->getVersion();
                            }
                        }
                        return true;
                    }
            );

            // Process Missing sections, creating a copyObject with the added cache for each.
            BaseType::processMissing(
                    range,
                    [&] (DataAccessRange missingRange) -> bool {
                        CopyObject *cpy = new CopyObject(missingRange, homeNode, 0);
                        if(homeNode == cache) {
                            cpy->setHomeNodeUpToDate(true);
                            cpy->removeCache(homeNode);
                        }
                        else
                            cpy->addCache(cache);
                        BaseType::insert(*cpy);
                        return true;
                    } 
                    );

        } else {
            // When CopyOut (incrementing version):
            // All regions are absorbed into the new one. 
            // Regions on the edges are shrunk (if needed).
            int version = 0;

            // Process all intersecting sections, resizing the edges as needed.
            BaseType::processIntersecting(
                    range,
                    [&] (CopySet::iterator position) -> bool {
                        if(position->getVersion() > version){
                            version = position->getVersion();
                        }

                        bool edge = false;	
                        if(position->getStartAddress() < range.getStartAddress()){
                            if(position->getEndAddress() > range.getEndAddress()){
                                CopyObject *cpy = new CopyObject(*position);
                                cpy->setStartAddress(range.getEndAddress() );
                                if(homeNode == cache) {
                                    cpy->setHomeNodeUpToDate(true);
                                    cpy->removeCache(homeNode);
                                }
                                BaseType::insert(*cpy);
                            }
                            position->setEndAddress(range.getStartAddress());
                            edge = true; 
                        }
                        if(position->getEndAddress() > range.getEndAddress()){
                            position->setStartAddress(range.getEndAddress());
                         edge = true;
                        }
                        if(!edge){
                            BaseType::erase(*position);
                            delete &(*position);
                        }

                        return true;
                    }
            );

            CopyObject *cpy = new CopyObject( range, homeNode, version + 1 );
            if(homeNode == cache) {
                cpy->setHomeNodeUpToDate(true);
                cpy->removeCache(homeNode);
            }
            else
                cpy->addCache(cache);
            BaseType::insert(*cpy);

            result = version + 1;
        }
    }

	return result;
}


CopySet::iterator CopySet::erase(void *address, int cache){

 	DataAccessRange eraseRange(address, address);	
	CopySet::iterator it = BaseType::find(eraseRange);
	it->removeCache(cache);
	if(!it->anyCache()){
		BaseType::erase(*it);
	}
	
	return it;	
}
