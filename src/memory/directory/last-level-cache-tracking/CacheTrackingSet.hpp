#ifndef CACHE_TRACKING_SET_HPP
#define CACHE_TRACKING_SET_HPP

#include <functional> // std::less
#include <atomic>

#include "IntrusiveLinearRegionTreap.hpp"
#include "CacheTrackingObject.hpp"
#include "CacheTrackingObjectLinkingArtifacts.hpp"
#include "IntrusiveLinearRegionTreapImplementation.hpp"


class CacheTrackingSet: public IntrusiveLinearRegionTreap<CacheTrackingObject, boost::intrusive::function_hook< CacheTrackingObjectLinkingArtifacts >, 
                               boost::intrusive::compare<std::less<CacheTrackingObjectKey> >,
                               boost::intrusive::priority<boost::intrusive::priority_compare<CacheTrackingObjectKey> > >{

private:
	typedef IntrusiveLinearRegionTreap<CacheTrackingObject, boost::intrusive::function_hook< CacheTrackingObjectLinkingArtifacts >, 
                                       boost::intrusive::compare<std::less<CacheTrackingObjectKey> >,
                                       boost::intrusive::priority<boost::intrusive::priority_compare<CacheTrackingObjectKey> > > BaseType;

    //! Size of the current working set of the last level cache.
    std::atomic<size_t> _currentWorkingSetSize;

    //! Counter to determine the last use of the accesses. Not thread-protected because it is not critical to have two accesses with the same lastUse.
    // TODO: IT MAY BE NEGATIVE IF WE USE MAX-HEAP
    long unsigned int _count; 


public:
	SpinLock _lock;

	CacheTrackingSet();

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
    void insert(DataAccessRange range, long unsigned int insertion_id); 

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

    /*! \brief Returns the size of the current working set.
     *
     *  Returns the size of all the accesses that are currently in the set.
     */
    inline std::size_t getCurrentWorkingSetSize() { return _currentWorkingSetSize; }

    /*! \brief Returns the free space in the last-level cache.
     *
     *  Returns the difference between the last-level cache size and the current working set size which is 
     *  the free space available in the last-level cache.
     */
    inline std::size_t getCurrentFreeSpace() { 
        return HardwareInfo::getLastLevelCacheSize() - _currentWorkingSetSize;
    }

    /*! \brief Returns an identifier to determine the last use of the accesses and increment the counter.
     *
     *  Returns an identifier which is used to determine the last use of an accesses and increment the counter 
     *  that provides the identifiers.
     */
    long unsigned int getLastUse() { return _count++; }
};

#include "CacheTrackingObjectLinkingArtifactsImplementation.hpp"
#endif //CACHE_TRACKING_SET_HPP
