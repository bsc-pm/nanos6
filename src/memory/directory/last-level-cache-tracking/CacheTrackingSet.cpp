#include "CacheTrackingSet.hpp"
#include <DataAccessRange.hpp>
#include <iostream>
#include <cstring>

#define _unused(x) ((void)(x))

CacheTrackingSet::CacheTrackingSet(): BaseType(), _count(0), _lock() {
    _currentWorkingSetSize = 0;
    _insertions = 0;
    _insertionsHint = 0;
    _pushBacks = 0;
    _perfectHints = 0;
}

CacheTrackingSet::iterator CacheTrackingSet::insert(DataAccessRange range, long unsigned int insertion_id, bool present, CacheTrackingSet::iterator &hint){
    Instrument::Timer aux;
    aux.start();
#ifndef NDEBUG
    checkRange(range);
#endif
    std::size_t cacheLineSize = HardwareInfo::getLastLevelCacheLineSize();
    _unused(cacheLineSize);
    uintptr_t start = (uintptr_t)range.getStartAddress();
    uintptr_t end = (uintptr_t)range.getEndAddress();
    std::size_t rangeSize = end - start;
    DataAccessRange tmp = DataAccessRange((void *)start, rangeSize);
    CacheTrackingObjectKey keyToInsert = CacheTrackingObjectKey(tmp, insertion_id);
    // TODO: Pool_alloc
    CacheTrackingObject * objectToInsert; // = new CacheTrackingObject(keyToInsert);
    CacheTrackingSet::iterator result = BaseType::end();

    if(present) {
        assert(hint==BaseType::find(keyToInsert));
        //! The hint passed is the iterator of the element we want.
        hint->updateLastUse(insertion_id);
        result = hint;
        //Instrument::Timer timerFind;
        //timerFind.start();
        //BaseType::iterator it = BaseType::find(keyToInsert);
        //timerFind.stop();
        //_timerFind+=timerFind;
        //if(it != BaseType::end()) {
        //    it->updateLastUse(insertion_id);
        //}
    }
    else {
        assert(!BaseType::contains(tmp));
        //std::cerr << "Inserting new range [" << (void *)start << ", " << (void *)end << "] WITHOUT overlappings." << std::endl;
        //! There is no overlapping.
        //std::cerr << "_currentWorkingSetSize: " << _currentWorkingSetSize << std::endl;

        //! Check if the access exceeds the total cache size
        if(rangeSize > getAvailableLastLevelCacheSize()) {
            //! The access does not fit into the cache even if it is empty. 
            //! Let's split the access to insert, at least, the part of the access that fits into the cache.

            //! Calculate the part that exceeds the cache size.
            std::size_t excess = rangeSize - getAvailableLastLevelCacheSize();
            assert(excess % cacheLineSize == 0);
            //! Advance the start in the excess so the access can fit in the cache.
            start += excess;
            //! Substract the excess from the access size.
            rangeSize -= excess;
            assert(rangeSize == (end-start));
            //! Update the access to be inserted discarding the excess.
            tmp = DataAccessRange((void *)start, rangeSize);
        }
        //! Check if the range fits into the cache.
        assert(rangeSize <= getAvailableLastLevelCacheSize());

        //! Construct the object to insert into the treap.
        keyToInsert = CacheTrackingObjectKey(tmp, insertion_id);
        objectToInsert = new CacheTrackingObject(keyToInsert);

        //! Check that the range is aligned to cache line.
#ifndef NDEBUG
        checkRange(objectToInsert->getAccessRange());
#endif

        //! Actually insert the object.
        Instrument::Timer timerActualInsert;
        timerActualInsert.start();
        if(hint != BaseType::end()) {
            //std::cerr << "Using hint (" << hint->getStartAddress() << ") for inserting: " << tmp.getStartAddress() << std::endl;
            //if(hint == BaseType::upper_bound(keyToInsert))
            //    _perfectHints++;
            //result = BaseType::insert(hint, *objectToInsert);
#ifndef NDEBUG
            //! Check that hint is actually the succesor of keyToInsert.
            assert(upper_bound(keyToInsert) == hint);
#endif
            Instrument::Timer timerInsertHint;
            timerInsertHint.start();
            result = BaseType::insert_before(hint, *objectToInsert);
            timerInsertHint.stop();
            _timerInsertsHint+=timerInsertHint;
            _insertionsHint++;
        }
        else {
            //! If hint is BaseType::end() it means either the treap is empty or the access goes to the end. Both cases allow a push_back.
            //result = BaseType::insert(*objectToInsert).first;
#ifndef NDEBUG
            //! Check that this is actually the greatest key.
            assert(upper_bound(keyToInsert) == BaseType::end());
#endif
            Instrument::Timer timerPushBack;
            timerPushBack.start();
            BaseType::push_back(*objectToInsert);
            timerPushBack.stop();
            _timerPushBacks+=timerPushBack;
            //! (BaseType::end()--) is the just pushed_back element.
            result--;
            _pushBacks++;
        }
        timerActualInsert.stop();
        _timerActualInsert+=timerActualInsert;

        //! Increment the current working set size.
        //std::cerr << "Incrementing _currentWorkingSetSize from " << _currentWorkingSetSize << " to " << _currentWorkingSetSize + rangeSize << std::endl;
        _currentWorkingSetSize += rangeSize;
    }
    _insertions++;
    //std::cerr << "Range [" << tmp.getStartAddress() << ", " << tmp.getEndAddress() << "] with lastUse " << insertion_id << " has been inserted." << std::endl;

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
    //! Check that there is some access in the treap as we have already inserted one.
    assert(BaseType::size() >= 1);
    aux.stop();
    _timer+=aux;
#ifndef NDEBUG
    //std::cerr << "CHECKING TREAP SIZE." << std::endl;
    std::size_t treapSize = 0;
    BaseType::processAll(
        [&] (BaseType::iterator it) -> bool {
            treapSize += it->getSize();
            checkRange(it->getAccessRange());
            return true;
        }
    );
    //std::cerr << "TREAP SIZE: " << treapSize << std::endl;
    //std::cerr << "CURRENT WORKING SET SIZE: " << _currentWorkingSetSize << std::endl;
    assert(treapSize == _currentWorkingSetSize);
#endif
    return result;
}

void CacheTrackingSet::evict() {
    if(_currentWorkingSetSize <= getAvailableLastLevelCacheSize())
        return;
    std::size_t size = _currentWorkingSetSize - getAvailableLastLevelCacheSize();
    std::size_t evictedSize = 0;
    while(size > 0) {
        //! Get LRU access.
        BaseType::iterator it = BaseType::top();
        assert(it != BaseType::end());
        if(it->getSize() > size) {
            uintptr_t newStartAddress = (uintptr_t) (it->getStartAddress())+size;
            it->setStartAddress((void *)newStartAddress);
            assert(it->getStartAddress() < it->getEndAddress() && "StartAddress cannot be greater than EndAddress");
            //std::cerr << "1.Decrementing _currentWorkingSetSize from " << _currentWorkingSetSize << " to " << _currentWorkingSetSize - size << std::endl;
            _currentWorkingSetSize -= size;
            evictedSize += size;
        }
        else {
            //std::cerr << "2.Decrementing _currentWorkingSetSize from " << _currentWorkingSetSize << " to " << _currentWorkingSetSize - it->getSize() << std::endl;
            _currentWorkingSetSize -= it->getSize();
            BaseType::erase(*it);
            evictedSize += it->getSize();
        }
        size = _currentWorkingSetSize - getAvailableLastLevelCacheSize();
    }
    //std::cerr << "EVICTED " << evictedSize << " BYTES." << std::endl;
    //std::cerr << "CURRENT WORKING SET SIZE: " << _currentWorkingSetSize << std::endl;
    assert(_currentWorkingSetSize <= getAvailableLastLevelCacheSize());
#ifndef NDEBUG
    //std::cerr << "CHECKING TREAP SIZE." << std::endl;
    std::size_t treapSize = 0;
    BaseType::processAll(
        [&] (BaseType::iterator it) -> bool {
            treapSize += it->getSize();
            checkRange(it->getAccessRange());
            return true;
        }
    );
    assert(treapSize == _currentWorkingSetSize);
#endif
}

void CacheTrackingSet::checkRange(DataAccessRange range) {
    std::size_t cacheLineSize = HardwareInfo::getLastLevelCacheLineSize();
    uintptr_t auxStart= (uintptr_t)range.getStartAddress();
    uintptr_t auxEnd = (uintptr_t)range.getEndAddress();
    std::size_t actualRangeSize = auxEnd - auxStart;
    _unused(cacheLineSize);
    _unused(actualRangeSize);
    assert(((auxStart % cacheLineSize == 0) && 
            (auxEnd % cacheLineSize == 0) && 
            (actualRangeSize % cacheLineSize == 0)) 
            && "Not fitting in cache line");
}
