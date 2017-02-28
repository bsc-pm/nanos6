#include "CacheTrackingSet.hpp"
#include <DataAccessRange.hpp>
#include <iostream>
#include <cstring>
#include <list>

CacheTrackingSet::CacheTrackingSet(): BaseType(), _count(0) {
    _currentWorkingSetSize = 0;
}

void CacheTrackingSet::insert(DataAccessRange range){
    //! Calculate actual range considering that HW caches works at line granularity.
    std::size_t cacheLineSize = HardwareInfo::getLastLevelCacheLineSize();
    std::size_t cacheSize = HardwareInfo::getLastLevelCacheSize();
    char * auxStart = (char *)range.getStartAddress();
    char * auxEnd = (char *)range.getEndAddress();
    char * actualStartAddress = (char*) (((std::size_t)auxStart/cacheLineSize)*cacheLineSize); 
    char * actualEndAddress = (char *) ((((std::size_t)auxEnd/cacheLineSize)+1)*cacheLineSize); 
    std::size_t actualRangeSize = actualEndAddress - actualStartAddress;
    DataAccessRange tmp = DataAccessRange(actualStartAddress, actualRangeSize);

    //! Check overlappings. Possible scenarios:
    //!     1. No overlapping.
    //!     2. Small new access contained in a big access already present in the set.
    //!         Split the big access.
    //!     3. Big new access which contains a small access already present in the set.
    //!         The new access is merged with the old one.
    //!     4. Partial overlapping between the new access and one already present in the set.
    //!         Split the old access and merge the overlapped part with the new one.
    //!     5. Identical access than one already present in the set.

    bool overlapping = false;
    BaseType::iterator it = BaseType::find(tmp);
    if(it != BaseType::end()) {
        //! Scenario 5.
        //! Access is already in the set. Just update its last use.
        it->updateLastUse(_count++);
        overlapping = true;
    }
    //else {
    //    if(range.fullyContainedIn(it->_range)) {
    //        //! Scenario 2.
    //        //! Scenario 2. Split the big access. The new access is updated to 
    //        //! have the lastUse corresponding to the insertion time, while the 
    //        //! the rest of the big access keeps the same lastUse.
    //    }
    //}

//        //! Access is not in the set yet. Check scenarios 1-4.
//        it = BaseType::begin();
//        std::list<BaseType::iterator> containedAccesses;
//        std::list<BaseType::iterator> partialOverlappedAccesses;
//        while(it != BaseType::end() && it->_range.getStartAddress() <= actualEndAddress) {
//            //! Advance iterator
//            BaseType::iterator aux = it;
//            it++;
//
//            //! If the current range ends before starting the new one, skip this access. 
//            if(it->_range.getEndAddress() <= actualStartAddress)
//                continue;
//
//            //! At this point, we know that new access is overlapping with the current access.
//            overlapping = true;
//            if(it->_range.getStartAddress() < actualStartAddress) {
//                //! Current access starts before new access. Check where does it end.
//                if(it->_range.getEndAddress() >= actualEndAddress) {
//                    //! Scenario 2. Split the big access. The new access is updated to 
//                    //! have the lastUse corresponding to the insertion time, while the 
//                    //! the rest of the big access keeps the same lastUse.
//                    CacheTrackingObject *a = new CacheTrackingObject(tmp);
//                    a->updateLastUse(_count++);
//                    BaseType::insert(*a);
//                    CacheTrackingObject prev = new CacheTrackingObject(DataAccessRange(it->_range.getStartAddress(), actualStartAddress));
//                    prev->updateLastUse(it->getLastUse());
//                    BaseType::insert(*prev);
//                    CacheTrackingObject post = new CacheTrackingObject(DataAccessRange(actualEndAddress, it->_range.getEndAddress()));
//                    post->updateLastUse(it->getLastUse());
//                    BaseType::insert(*post);
//
//                    //! Incrementing the current working set size is not needed because 
//                    //! the new access is contained in a previous one.
//                    break;
//                }
//                else {
//                    //! Scenario 4. Split the old access and merge the overlapped part 
//                    //! with the new access to have the lastUse corresponding to the 
//                    //! insertion time, while the rest of the old access keeps the same 
//                    //! last use. We add the current access to a list to be processed 
//                    //! after the loop, otherwise it would invalidate the iterator.
//                    partialOverlappedAccesses.push_back(it);
//
//                    //! Increment the current working set size.
//                    std::size_t increment = (std::size_t)(actualEndAddress - (char *)it->_range.getEndAddress());
//                    _currentWorkingSetSize += increment;
//                }
//            }
//            else if(it->_range.getStartAddress() > actualStartAddress){
//                //! Current access starts after/inside new access. Check where does it end.
//                if(it->_range.getEndAddress() <= actualEndAddress) {
//                    //! Scenario 3. Replace the old access with the new one.
//
//                    //! Increment the current working set size.
//                }
//                else {
//                    //! Scenario 4. Split the old access and merge the overlapped part 
//                    //! with the new access to have the lastUse corresponding to the 
//                    //! insertion time, while the rest of the old access keeps the same 
//                    //! lastUse.
//
//                    //! Increment the current working set size.
//                }
//            }
//            else {
//                //! Same startAddress for new and current access.
//                //! We already know that endAddress is not the same, otherwise
//                //! it would be the same access and the find method would have 
//                //! found it.
//                if(it->range.getEndAddress() < actualEndAddress) {
//                    //! Scenario 3. Replace the old access with the new one.
//
//                    //! Increment the current working set size.
//                }
//                else {
//                    //! Scenario 2. Split the big access. The new access is updated to 
//                    //! have the lastUse corresponding to the insertion time, while the 
//                    //! the rest of the big access keeps the same lastUse.
//
//                    //! Increment the current working set size.
//                    //! Incrementing the current working set size is not needed because 
//                    //! the new access is contained in a previous one.
//                    break;
//                }
//            }
//
//        }
//    }
//
//    if(!overlapping) {
//        //! Scenario 1. Insert a new node.
//        CacheTrackingObject *a = new CacheTrackingObject(tmp);
//        a->updateLastUse(_count++);
//        BaseType::insert(*a);
//        //! Increment the current working set size.
//        _currentWorkingSetSize += actualRangeSize;
//    }
//
//    assert(_currentWorkingSetSize <= cacheSize && "Current working set size must not be greater than cache size.");
}

void CacheTrackingSet::evict(){
    //! Get LRU access.
    BaseType::iterator erase = BaseType::top();
    CacheTrackingObject &aux = *erase;
    ////! Decrement current working set size.
    _currentWorkingSetSize -= erase->_range.getSize(); 
    ////! Remove the access from the set.
    BaseType::erase(aux);
}

