#ifndef DIRECTORY_HPP
#define DIRECTORY_HPP

#include <TaskDataAccesses.hpp>

#include "lowlevel/SpinLock.hpp"
#include "copies/CopySet.hpp"
#include "pages/MemoryPageSet.hpp"
#include "last-level-cache-tracking/CacheTrackingSet.hpp"

//#include <vector>
//#include <unordered_map>

class Directory {


private:
    //! Members for debug purposes
    Instrument::Timer _timer;
    Instrument::Timer _timerCacheTrackingSetLock;
    Instrument::Timer _timerTaskDataAccessesLock;
    Instrument::Timer _timerProcess;
    Instrument::Timer _timerFragment;
    Instrument::Timer _timerIntersectingAndMissing;
    Instrument::Timer _timerIntersecting;
    Instrument::Timer _timerMissing;
    Instrument::Timer _timerEvict;
    Instrument::Timer _timerUpdate;
    Instrument::Timer _timerErase;
    Instrument::Timer _timerComputeScore;
    long unsigned int _tasksRegistered;
    long unsigned int _tasksComputed;

	SpinLock _lock;

    /* Tracks copies of software managed caches of the different devices 
       It is only required when using software managed caches.
     */
    bool _enableCopies;
	CopySet _copies;

    /* Tracks the homeNode of each access. 
     */
    bool _enableHomeNodeTracking;
	MemoryPageSet _pages;	

    /* Tracks, for each NUMA node, the current working set of the last 
       level cache.
     */
    //std::unordered_map<unsigned int, CacheTrackingSet *> _lastLevelCacheTracking;
    //std::vector<CacheTrackingSet *> _lastLevelCacheTracking;
    bool _enableLastLevelCacheTracking;
    CacheTrackingSet ** _lastLevelCacheTracking;

	static Directory *_instance;

	Directory();
public:	
	
	/*! Initializes the directory */
	static void initialize();
	
	/*! Delete method for the directory */
	static void dispose();

	/*!	\brief Returns the version of a copy or -1 if it is not present 
	 *	
	 *	Return the last version of a copy object.
	 *	If the copy is not on the directory a -1 is returned instead.
	 *
	 *	\param address The starting address of the copy
	 *	\param size The size of the copy
	 */
	static int getVersion(void *address);

	/*! \brief Registers a region that has been copied at a certain cache 
	 * 
	 *  \param address The base address of the copied region
	 *  \param size The size of the copied region
     *  \param homeNode The homeNode of the copied region
	 *  \param cache The cache to which the region is copied
	 *  \param increment True if the version needs to be incremented
	 */
	static int insertCopy(void *address, size_t size, int homeNode, int cache, bool increment);

	/*! \brief Registers a region that has been removed from a cache
	 *  
	 *  \param address The base address of the evicted region
	 *	\param size Size of the evicted region
	 *	\param cache The cache from which the region is evicted
	 */
	static void eraseCopy(void *address, int cache);

	/*! \brief Retrieves location data of the data accesses of a task in order to determine execution place.
	 *	
	 *  Accesses the information in the directory in order to determine in which NUMA node will a task be executed.
	 *	First it accesses the home node information of each access and registers it inside the access. 
	 *	If the region is not registered move_pages will be called to retrieve the information.
     *  It returns a vector (for the time being, a vector is enough, especially taking into account that the size 
     *  of this vector is the number of NUMA nodes which usually is not a very big number) with a score for each NUMA node.
	 *	
	 *	\param accesses Data accesses of a Task
	 *
	 */
	static std::vector<double> computeNUMANodeAffinity(TaskDataAccesses &accesses);	

    /*! \brief Returns a bitset indicating which caches have the dataAccess in its last version
     *  \param (in) address The startAddress of the dataAccess
     */
    static cache_mask getCaches(void *address);

    /*! \brief Returns the homeNode of a region
     *  \param (in) address The startAddress of the region.
     */
    static int getHomeNode(void *address);

    /*! \brief Returns a boolean indicating whether the homeNode is up to date.
      * \param (in) address The startAddress of the region.
      */
    static bool isHomeNodeUpToDate(void *address); 
    
    /*! \brief Set a boolean indicating whether the homeNode is up to date.
      * \param (in) address The start address of the region.
      * \param (in) b The boolean to set.
      */
    static void setHomeNodeUpToDate(void *address, bool b);

    static void createLastLevelCacheTracking(unsigned int nodes);
    static void registerLastLevelCacheData(TaskDataAccesses &accesses, unsigned int NUMANodeId, Task * task); 
    static double computeTaskAffinity(Task * task, unsigned int NUMANodeId);
};

#endif //DIRECTORY_HPP
