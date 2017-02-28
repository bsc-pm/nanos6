#include "CacheTrackingSet.hpp"
#include <DataAccessRange.hpp>
#include <iostream>
#include <cstring>
#include <list>

#define _unused(x) ((void)(x))

void checkRange(DataAccessRange range);

CacheTrackingSet::CacheTrackingSet(): BaseType(), _count(0), _lock() {
    _currentWorkingSetSize = 0;
}

void CacheTrackingSet::insert(DataAccessRange range, long unsigned int insertion_id){
    //! Calculate actual range considering that HW caches works at line granularity.
    std::size_t cacheLineSize = HardwareInfo::getLastLevelCacheLineSize();
    char * auxStart = (char *)range.getStartAddress();
    char * auxEnd = (char *)range.getEndAddress();
    char * actualStartAddress = (char*) (((std::size_t)auxStart/cacheLineSize)*cacheLineSize); 
    char * actualEndAddress = (char *) ((((std::size_t)auxEnd/cacheLineSize)+1)*cacheLineSize); 
    std::size_t actualRangeSize = actualEndAddress - actualStartAddress;
    DataAccessRange tmp = DataAccessRange(actualStartAddress, actualRangeSize);
    CacheTrackingObjectKey keyToInsert = CacheTrackingObjectKey(tmp, insertion_id);
    CacheTrackingObject * objectToInsert = new CacheTrackingObject(keyToInsert);

    //! Check overlappings. Possible scenarios:
    //!     1. No overlapping.
    //!     2. Small new access contained in a big access already present in the set.
    //!         Split the big access.
    //!     3. Big new access which contains a small access already present in the set.
    //!         The new access is merged with the old one.
    //!     4. Partial overlapping between the new access and one already present in the set.
    //!         Split the old access and merge the overlapped part with the new one.
    //!     5. Identical access than one already present in the set.


    //bool overlapping = false;
    BaseType::iterator it = BaseType::find(keyToInsert);
    if(it != BaseType::end()) {
        //! Scenario 5.
        //! Access is already in the set. Just update its last use.
        it->updateLastUse(insertion_id);
        //overlapping = true;
    }
    else {
        if(!BaseType::contains(tmp)) {
            //std::cerr << "Inserting new range [" << (void *)actualStartAddress << ", " << (void *)actualEndAddress << "] WITHOUT overlappings." << std::endl;
            //! Scenario 1. There is no overlapping.
            //BaseType::insert(keyToInsert);
            checkRange(objectToInsert->getAccessRange());
            BaseType::insert(*objectToInsert);
            //std::cerr << "Increment: " << actualRangeSize << std::endl;
            //std::cerr << "Free space: " << getCurrentFreeSpace() << std::endl;
            //std::cerr << "_currentWorkingSetSize: " << _currentWorkingSetSize << std::endl;
            while(actualRangeSize > getCurrentFreeSpace()) {
                //std::cerr << "Evicting because " << actualRangeSize << " bytes are needed and only " << getCurrentFreeSpace() 
                //    << " remains free in the cache" << std::endl;
                //! Evict until all the data of the task fits into the cache. 
                evict();
            }
            //! Increment the current working set size.
            _currentWorkingSetSize += actualRangeSize;
            return;   
        }
        //std::cerr << "Inserting new range [" << (void *)actualStartAddress << ", " << (void *)actualEndAddress << "] WITH overlappings." << std::endl;
        std::size_t increment = actualRangeSize;
        BaseType::processIntersecting(
            tmp,
            [&] (CacheTrackingSet::iterator &intersectingAccess) -> bool {
                if(tmp.fullyContainedIn(intersectingAccess->getAccessRange())) {
                    //! Scenario 2. Split the big access. The new access is updated to 
                    //! have the lastUse corresponding to the insertion time, while the 
                    //! the rest of the big access keeps the same lastUse.
                    if(tmp.getStartAddress() > intersectingAccess->getStartAddress()) {
                        intersectingAccess->setEndAddress(tmp.getStartAddress());
                    }
                    else {
                        intersectingAccess->setStartAddress(tmp.getEndAddress());
                    }
                    //! Incrementing the current working set size is not needed because 
                    //! the new access is contained in a previous one.
                    increment = 0;
                }
                else {
                    if(intersectingAccess->getAccessRange().fullyContainedIn(tmp)) {
                        //! Scenario 3. Erase the small access because the new one contains it.
                        increment -= intersectingAccess->getSize();
                        BaseType::erase(*intersectingAccess);
                    }
                    else {
                        //! Scenario 4. Split the old access to be only the non-overlapping part.
                        if(tmp.getStartAddress() < intersectingAccess->getStartAddress()) {
                            increment -= ((char *)tmp.getEndAddress() - (char *)intersectingAccess->getStartAddress());
                            intersectingAccess->setStartAddress(tmp.getEndAddress());
                        }
                        else {
                            increment -= ((char *)intersectingAccess->getEndAddress() - (char *)tmp.getStartAddress());
                            intersectingAccess->setEndAddress(tmp.getStartAddress());
                        }
                    }
                }
                return true;
            }
        );	
        checkRange(objectToInsert->getAccessRange());
        BaseType::insert(*objectToInsert);
        //! Evict if needed. Evictions can be done after insertions because we are actually doing only a track of the information that would be there.
        //! It should be enough to check if the size of the new inserted data (increment) is lower or equal to the free space.
        //while(_currentWorkingSetSize >= HardwareInfo::getLastLevelCacheSize()) {
        //std::cerr << "Increment: " << increment << std::endl;
        //std::cerr << "Free space: " << getCurrentFreeSpace() << std::endl;
        //std::cerr << "_currentWorkingSetSize: " << _currentWorkingSetSize << std::endl;
        while(increment > getCurrentFreeSpace()) {
            //std::cerr << "Evicting because " << increment << " bytes are needed and only " << getCurrentFreeSpace() 
            //          << " remains free in the cache" << std::endl;
            //! Evict until all the data of the task fits into the cache. 
            evict();
        }
        //! Increment the current working set size.
        _currentWorkingSetSize += increment;
    }

    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "Listing last level cache tracking set." << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //BaseType::processAll(
    //    [&] (BaseType::iterator it) -> bool {
    //        std::cerr << "Range [" << it->getStartAddress() << ", " << it->getEndAddress() << "] with lastUse " << it->getLastUse() << "." << std::endl;
    //        return true;
    //    }
    //);
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    assert(_currentWorkingSetSize <= (std::size_t) HardwareInfo::getLastLevelCacheSize() && "Current working set size must not be greater than cache size.");
    // FIXME: TEMPORAL WORKAROUND
    //assert((_currentWorkingSetSize-cacheLineSize) <= (std::size_t) HardwareInfo::getLastLevelCacheSize() && "Current working set size must not be greater than cache size.");
}

void CacheTrackingSet::evict(){
    // TODO: Think about a mechanism to prevent evicting data that may be in use right now. 
    // Maybe, set a threshold (e.g. 10) to prevent evicting data that its lastUse difference 
    // with _count is lower than the threshold.
    //! Get LRU access.
    BaseType::iterator erase = BaseType::top();
    CacheTrackingObject &aux = *erase;
    ////! Decrement current working set size.
    _currentWorkingSetSize -= erase->_key._range.getSize(); 
    ////! Remove the access from the set.
    BaseType::erase(aux);
}

void checkRange(DataAccessRange range) {
    std::size_t cacheLineSize = HardwareInfo::getLastLevelCacheLineSize();
    char * auxStart = (char *)range.getStartAddress();
    char * auxEnd = (char *)range.getEndAddress();
    std::size_t actualRangeSize = auxEnd - auxStart;
    _unused(cacheLineSize);
    _unused(actualRangeSize);
    assert((((std::size_t)auxStart % cacheLineSize == 0) && 
            ((std::size_t)auxEnd % cacheLineSize == 0) && 
            (actualRangeSize % cacheLineSize == 0)) 
            && "Not fitting in cache line");
}
