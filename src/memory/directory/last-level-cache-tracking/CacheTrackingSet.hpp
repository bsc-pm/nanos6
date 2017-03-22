#ifndef CACHE_TRACKING_SET_HPP
#define CACHE_TRACKING_SET_HPP

#include <functional> // std::less
#include <atomic>
#include <set>

#include "IntrusiveLinearRegionTreap.hpp"
#include "CacheTrackingObject.hpp"
#include "CacheTrackingObjectLinkingArtifacts.hpp"
#include "IntrusiveLinearRegionTreapImplementation.hpp"
#include "instrument/stats/Timer.hpp"
#include "lowlevel/RWSpinLock.hpp"

#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define round_up(x, y) ((((x)-1) | __round_mask(x, y))+1)
#define round_down(x, y) ((x) & ~__round_mask(x, y))

class CacheTrackingSet: public IntrusiveLinearRegionTreap<CacheTrackingObject, boost::intrusive::function_hook< CacheTrackingObjectLinkingArtifacts >, 
                               boost::intrusive::compare<std::less<CacheTrackingObjectKey> >,
                               boost::intrusive::priority<boost::intrusive::priority_compare<CacheTrackingObjectKey> > >{

private:
	typedef IntrusiveLinearRegionTreap<CacheTrackingObject, boost::intrusive::function_hook< CacheTrackingObjectLinkingArtifacts >, 
                                       boost::intrusive::compare<std::less<CacheTrackingObjectKey> >,
                                       boost::intrusive::priority<boost::intrusive::priority_compare<CacheTrackingObjectKey> > > BaseType;

    //! Size of the current working set of the last level cache.
    std::atomic<std::size_t> _currentWorkingSetSize;

    //! Counter to determine the last use of the accesses. Not thread-protected because it is not critical to have two accesses with the same lastUse.
    long unsigned int _count; 

    //! Members for debug purposes
    long unsigned int _insertions;
    long unsigned int _insertionsHint;
    long unsigned int _pushBacks;
    long unsigned int _perfectHints;
    Instrument::Timer _timer;
    Instrument::Timer _timerActualInsert;
    Instrument::Timer _timerFind;
    Instrument::Timer _timerInsertsHint;
    Instrument::Timer _timerPushBacks;

public:
	RWSpinLock _lock;

	CacheTrackingSet();
    ~CacheTrackingSet() {
        std::cerr << "Time in insert: " << _timer << " ns." << std::endl;
        std::cerr << "Time in actual insert: " << _timerActualInsert << " ns." << std::endl;
        std::cerr << "Time in inserts with hint: " << _timerInsertsHint << " ns." << std::endl;
        std::cerr << "Time in inserts with push_back: " << _timerPushBacks << " ns." << std::endl;
        std::cerr << "Time in find: " << _timerFind << " ns." << std::endl;
        std::cerr << "Total insertions: " << _insertions << "." << std::endl;
        std::cerr << "Total insertions with hint: " << _insertionsHint << "." << std::endl;
        std::cerr << "Perfect hints: " << _perfectHints << "." << std::endl;
        std::cerr << "Total push backs: " << _pushBacks << "." << std::endl;
        std::cerr << "Time per insertion: " << (double)(_timer/_insertions) << " ns." << std::endl;
        std::cerr << "Time per actual insertion: " << (double)(_timerActualInsert/_insertions) << " ns." << std::endl;
        std::cerr << "Time per actual insertion using hint: " << (double)(_timerInsertsHint/_insertionsHint) << " ns." << std::endl;
        std::cerr << "Time per actual insertion using push_back: " << (double)(_timerPushBacks/_pushBacks) << " ns." << std::endl;
        std::cerr << "Time in find per insertion: " << (double)(_timerFind/_insertions) << " ns." << std::endl;
    }

    /*! \brief Inserts a new access in the last level cache tracking and updates its last use.
     *
     *  A new access is inserted in the set if it is not already present. 
     *  Either it is already present or it is inserted, the last use of the 
     *  access is updated with insertion_id.
     *  _currentWorkingSetSize is also updated with the size of the access.
     *
     *  Note that this is a HW cache tracking, so this tracking works with linesize.
     *  That means that the _currentWorkingSetSize will be updated in 
     *  ((int)(access_size/linesize)+1)*linesize. The range of the tracked access will be
     *  start_address = (int)(range_start_address/linesize)*linesize 
     *  end_address = ((int)(range_end_address/linesize)+1)*linesize.
     *
     *  \param range Range of the access to be inserted.
     *  \param insertion_id The lastUse of the access.
     */
    CacheTrackingSet::iterator insert(DataAccessRange range, long unsigned int insertion_id, bool present, CacheTrackingSet::iterator &hint); 

    /*! \brief Perform the evictions required until the _currentWorkingSetSize is equal or smaller  
     *         than the available cache size.
     *
     *  Look for the least recently used accesses and erase them until _currentWorkingSetSize is 
     *  equal or smaller than the available cache size.
     */
    void evict();

    /*! \brief Returns the size of the current working set.
     *
     *  Returns the size of all the accesses that are currently in the set.
     */
    inline std::size_t getCurrentWorkingSetSize() { return _currentWorkingSetSize; }

    /*! \brief Updates the size of the current working set.
     *
     *  Updates the size that is stored currently in the set.
     */
    inline void updateCurrentWorkingSetSize(long int update) { 
        assert((long int)(_currentWorkingSetSize + update) >= 0);
        _currentWorkingSetSize += update; 
    }

    /*! \brief Returns the free space in the last-level cache.
     *
     *  Returns the difference between the last-level cache size and the current working set size which is 
     *  the free space available in the last-level cache.
     */
    inline std::size_t getCurrentFreeSpace() { 
        std::size_t availableLastLevelCacheSize = getAvailableLastLevelCacheSize();
        //assert(availableLastLevelCacheSize >= _currentWorkingSetSize);
        std::size_t freeSpace = availableLastLevelCacheSize - _currentWorkingSetSize;
        return freeSpace;
    }

    /*! \brief Returns the available size of the last-level cache.
     *
     *  The cache, apart from the user data, contains the runtime internals, so we must 
     *  reserve some space for the internals. This method returns the size available for 
     *  user data, substracting the reserved space for runtime internals.
     */
    inline std::size_t getAvailableLastLevelCacheSize() {
        //! We must take into account that our internal data structures are also located 
        //! in the cache, so we should reserve some space. This must be adjusted to get 
        //! the best performance. For the time being, a 10% of the capacity will be reserved 
        //! for our internals.
        double reservedPart = 0.1;
        std::size_t availableSize = (1-reservedPart) * HardwareInfo::getLastLevelCacheSize();
        availableSize = round_down(availableSize, HardwareInfo::getLastLevelCacheLineSize());
        assert((availableSize <= HardwareInfo::getLastLevelCacheSize()) && 
               (availableSize % HardwareInfo::getLastLevelCacheLineSize() == 0));
        return availableSize;
    }

    /*! \brief Returns an identifier to determine the last use of the accesses and increment the counter.
     *
     *  Returns an identifier which is used to determine the last use of an accesses and increment the counter 
     *  that provides the identifiers.
     */
    long unsigned int getLastUse() { return _count++; }

    /*! \brief Checks if the range is aligned to cache line size. If it is not, it asserts, that's why the method
     *         is void.
     */
    static void checkRange(DataAccessRange);
};

#include "CacheTrackingObjectLinkingArtifactsImplementation.hpp"
#endif //CACHE_TRACKING_SET_HPP
