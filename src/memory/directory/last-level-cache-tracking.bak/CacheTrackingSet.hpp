#ifndef CACHE_TRACKING_SET_HPP
#define CACHE_TRACKING_SET_HPP

#include <functional> // std::less
#include <atomic>

#include "IntrusiveLinearRegionTreap.hpp"
#include "CacheTrackingObject.hpp"

//typedef boost::intrusive::treap_set< CacheTrackingObject, boost::intrusive::compare<std::greater<CacheTrackingObject> > > IntrusiveCacheTrackingTreap ;

class CacheTrackingSet: public IntrusiveLinearRegionTreap<CacheTrackingObject, 
                               boost::intrusive::compare<std::greater<CacheTrackingObject> >/*,
                               boost::intrusive::priority<CacheTrackingObject::default_priority>*/ >{

private:
	typedef IntrusiveLinearRegionTreap<CacheTrackingObject, 
                                       boost::intrusive::compare<std::greater<CacheTrackingObject> >/*,
                                       boost::intrusive::priority<CacheTrackingObject::default_priority>*/ > BaseType;

    //! Size of the current working set of the last level cache.
    std::atomic<size_t> _currentWorkingSetSize;

    //! Counter to determine the last use of the accesses. Not thread-protected because it is not critical to have two accesses with the same lastUse.
    // TODO: IT MAY BE NEGATIVE IF WE USE MAX-HEAP
    long unsigned int _count; 

public:

	CacheTrackingSet();

    /*! \brief Inserts a new access in the last level cache tracking and updates its last use.
     *
     *  A new access is inserted in the set if it is not already present. 
     *  Either it is already present or it is inserted, the last use of the 
     *  access is updated.
     *  _currentWorkingSetSize is also updated with the size of the access.
     *
     *  Note that this is a HW cache tracking, so this tracking works with linesize.
     *  That means that the _currentWorkingSetSize will be updated in 
     *  ((int)(access_size/linesize)+1)*linesize. The range of the tracked access will be
     *  start_address = (int)(range_start_address/linesize)*linesize 
     *  end_address = ((int)(range_end_address/linesize)+1)*linesize.
     *
     *  \param range Range of the access to be inserted.
     */
    void insert(DataAccessRange range); 

    /*! \brief The least recently used access is evicted. 
     *
     *  Look for the least recently used access and decrement _currentWorkingSetSize 
     *  in the size of the evicted access.
     *
     *  For the time being, accesses are not removed from the set unless the
     *  total number of CacheTrackingObjects is above a given threshold related 
     *  to the size of the cache that is being tracked.
     *
     */
    void evict();

    inline std::size_t getCurrentWorkingSetSize() { return _currentWorkingSetSize; }
};

#endif //CACHE_TRACKING_SET_HPP
