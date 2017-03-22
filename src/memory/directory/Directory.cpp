#include <mutex>

#include "Directory.hpp"
#include <TaskDataAccessesImplementation.hpp>
#include <iostream>

#define _unused(x) ((void)(x))

Directory *Directory::_instance = nullptr;

Directory::Directory(): _lock(), _copies(), _pages(){
}

void Directory::initialize(){
	Directory::_instance = new Directory();
	EnvironmentVariable<std::string> schedulerName("NANOS6_SCHEDULER", "default");
	if (schedulerName.getValue() == "locality") {
        //_instance->_enableHomeNodeTracking = true;
        _instance->_enableHomeNodeTracking = false;
        _instance->_enableLastLevelCacheTracking = true;
        //_instance->_enableLastLevelCacheTracking = false;
    }
    else {
        _instance->_enableHomeNodeTracking = false;
        _instance->_enableLastLevelCacheTracking = false;
    }
    _instance->_enableCopies = false;
    _instance->_tasksRegistered = 0;
    _instance->_tasksComputed = 0;
}

void Directory::dispose(){
    std::cerr << "Time in registerLastLevelCacheData: " << _instance->_timer << " ns." << std::endl; 
    std::cerr << "Time in TaskDataAccesses.procesAll: " << _instance->_timerProcess << " ns." << std::endl;
    std::cerr << "Time in processIntersectingAndMissing: " << _instance->_timerIntersectingAndMissing << " ns." << std::endl;
    std::cerr << "Time in intersecting: " << _instance->_timerIntersecting << " ns." << std::endl;
    std::cerr << "Time in missing: " << _instance->_timerMissing << " ns." << std::endl;
    std::cerr << "Time in erase: " << _instance->_timerErase << " ns." << std::endl;
    std::cerr << "Time in update: " << _instance->_timerUpdate << " ns." << std::endl;
    std::cerr << "Time in fragmentByIntersection: " << _instance->_timerFragment << " ns." << std::endl;
    std::cerr << "Time waiting for TaskDataAccesses lock: " << _instance->_timerTaskDataAccessesLock << " ns." << std::endl;
    std::cerr << "Time waiting for CacheTrackingSet lock: " << _instance->_timerCacheTrackingSetLock << " ns." << std::endl;
    std::cerr << "Time evicting: " << _instance->_timerEvict << " ns." << std::endl;
    std::cerr << "Total registered tasks: " << _instance->_tasksRegistered << "." << std::endl;
    std::cerr << "Time registering accesses per task: " << (double)(_instance->_timer/_instance->_tasksRegistered) << " ns." << std::endl;
    std::cerr << "Time in computeTaskAffinity: " << _instance->_timerComputeScore << " ns." << std::endl;
    std::cerr << "Total number of tasks processed in computeTaskAffinity: " << _instance->_tasksComputed << "." << std::endl;
    if(_instance->_tasksComputed != 0)
        std::cerr << "Time in computeTaskAffinity per task: " << (double)(_instance->_timerComputeScore/_instance->_tasksComputed) << " ns." << std::endl;
    if(_instance->_enableLastLevelCacheTracking) {
        unsigned int nodes = HardwareInfo::getMemoryNodeCount();
        for(unsigned int i = 0; i < nodes; i++) 
            delete _instance->_lastLevelCacheTracking[i];
        free(_instance->_lastLevelCacheTracking);
    }
	delete Directory::_instance;
}

int Directory::getVersion(void *address){
    assert(_instance->_enableCopies);
	CopySet::iterator it = Directory::_instance->_copies.find(address);
	if(it != Directory::_instance->_copies.end()){
		return it->getVersion();
	} else {
		return -1;
	}
}

int Directory::insertCopy(void *address, size_t size, int homeNode, int cache, bool increment){
    assert(_instance->_enableCopies);
	std::lock_guard<SpinLock> guard(Directory::_instance->_lock);
	return Directory::_instance->_copies.insert(address, size, homeNode, cache, increment);
}

void Directory::eraseCopy(void *address, int cache){
    assert(_instance->_enableCopies);
	std::lock_guard<SpinLock> guard(Directory::_instance->_lock);
	Directory::_instance->_copies.erase(address, cache);
}

std::vector<double>  Directory::computeNUMANodeAffinity(TaskDataAccesses &accesses ){
    assert(_instance->_enableHomeNodeTracking);
	std::lock_guard<TaskDataAccesses::spinlock_t> guardAccesses(accesses._lock);
	std::lock_guard<SpinLock> guardDirectory(Directory::_instance->_lock);

    //! Create result vector initializing all scores to 0.
    std::vector<double> result(HardwareInfo::getMemoryNodeCount(), 0);
	
	//! Process all Data accesses
	accesses._accesses.processAll(
		[&] ( TaskDataAccesses::accesses_t::iterator it ) -> bool {

			//! Process all possible gaps in the pages directory and insert them in the pages list
			Directory::_instance->_pages.processMissing(
				it->getAccessRange(),
				[&] (DataAccessRange missingRange) -> bool {
					Directory::_instance->_pages.insert(missingRange);
					return true;
				}
			);	

			//! Search for all pages in the pages list
			Directory::_instance->_pages.processIntersecting(
				it->getAccessRange(),
				[&] (MemoryPageSet::iterator position) -> bool {
                    assert(position->getLocation() >= 0 && "Wrong location");
					it->_homeNode = position->getLocation();
                    //! Double weight for accesses which require writing.
                    double score = (it->_type != READ_ACCESS_TYPE) ? 2*it->getAccessRange().getSize() : it->getAccessRange().getSize();
                    result[it->_homeNode] += score;
					return true; 
				}
			);
			return true;
		} 
	);	
    return result;
}

cache_mask Directory::getCaches(void *address) {
    assert(_instance->_enableCopies);
	std::lock_guard<SpinLock> guard(Directory::_instance->_lock);
	CopySet::iterator it = Directory::_instance->_copies.find(address);
    assert(it != _instance->_copies.end() && "The copy must be in the directory");
    return it->getCaches();
}

int Directory::getHomeNode(void *address) {
    assert(_instance->_enableCopies);
	std::lock_guard<SpinLock> guard(Directory::_instance->_lock);
    CopySet::iterator it = _instance->_copies.find(address);
    if(it != _instance->_copies.end()) {
        return it->getHomeNode();
    }
    else {
        return -1;
    }
}

bool Directory::isHomeNodeUpToDate(void *address) {
    assert(_instance->_enableCopies);
	std::lock_guard<SpinLock> guard(Directory::_instance->_lock);
    CopySet::iterator it = _instance->_copies.find(address);
    if(it != _instance->_copies.end()) {
        return it->isHomeNodeUpToDate();
    }
    else {
        //! If the copy is not in the directory yet, it is up to date in the Directory.
        return true;
    }
}

void Directory::setHomeNodeUpToDate(void *address, bool b) {
    assert(_instance->_enableCopies);
	std::lock_guard<SpinLock> guard(Directory::_instance->_lock);
    CopySet::iterator it = _instance->_copies.find(address);
    //assert(it != _instance->_copies.end() && "The copy must be in the directory");
    if(it != _instance->_copies.end()) {
        it->setHomeNodeUpToDate(b);
    }
}

void Directory::createLastLevelCacheTracking(unsigned int nodes) {
    if(!_instance->_enableLastLevelCacheTracking)
        return;

    _instance->_lastLevelCacheTracking = (CacheTrackingSet **) calloc(nodes, sizeof(CacheTrackingSet *));
    //_instance->_lastLevelCacheTracking.reserve(nodes);
    for(unsigned int i = 0; i < nodes; i++) {
        CacheTrackingSet *a = new CacheTrackingSet();
        _instance->_lastLevelCacheTracking[i] = a;
    }
}

void Directory::registerLastLevelCacheData(TaskDataAccesses &accesses, unsigned int NUMANodeId, Task * task) {
    _instance->_tasksRegistered++;
    Instrument::Timer aux;
    aux.start();
    if(!_instance->_enableLastLevelCacheTracking) {
        aux.stop();
        _instance->_timer+=aux;
        return;
    }

    _unused(task);
    CacheTrackingSet &cacheTracking = *(_instance->_lastLevelCacheTracking[NUMANodeId]);
    long unsigned int lastUse = cacheTracking.getLastUse();
    std::size_t cacheLineSize = HardwareInfo::getLastLevelCacheLineSize();
	//std::lock_guard<TaskDataAccesses::spinlock_t> guardAccesses(accesses._lock);
    Instrument::Timer timerTaskDataAccessesLock;
    timerTaskDataAccessesLock.start();
    accesses._lock.lock();
    timerTaskDataAccessesLock.stop();
    _instance->_timerTaskDataAccessesLock+=timerTaskDataAccessesLock;
    //std::lock_guard<SpinLock> guardCache(cacheTracking._lock);
    Instrument::Timer aux2;
    aux2.start();
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "Registering last level cache data of task " << task->getTaskInfo()->task_label << " (" << task << ") with task data size " 
    //          << task->getDataSize() << ". The number of accesses is " << accesses._accesses.size() <<  "." << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "Listing last level cache tracking set registerLastLevelCacheData start." << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //cacheTracking.processAll(
    //    [&] (CacheTrackingSet::iterator it) -> bool {
    //        std::cerr << "Range [" << it->getStartAddress() << ", " << it->getEndAddress() << "] with lastuse " << it->getLastUse() << "." << std::endl;
    //        return true;
    //    }
    //);
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    Instrument::Timer timerProcess;
    timerProcess.start();
    TaskDataAccesses::accesses_t collapsedAccesses;
    void * collapsedStart = nullptr;
    void * collapsedEnd = nullptr;
    accesses._accesses.processAll(
        [&] (TaskDataAccesses::accesses_t::iterator it) -> bool {
            uintptr_t roundedStart = (uintptr_t)it->getAccessRange().getStartAddress();
            uintptr_t roundedEnd = (uintptr_t)it->getAccessRange().getEndAddress();
            roundedStart = round_down(roundedStart, cacheLineSize);
            roundedEnd = round_up(roundedEnd, cacheLineSize);
            //std::cerr << "Proccessing range [" << roundedStart << ", " << roundedEnd << "." << std::endl;
            if(it == accesses._accesses.begin())
                collapsedStart = (void *)roundedStart;
            if((roundedStart > (uintptr_t)collapsedEnd) && (it != accesses._accesses.begin())) {
                DataAccessRange tmp = DataAccessRange(collapsedStart, collapsedEnd);
#ifndef NDEBUG
                CacheTrackingSet::checkRange(tmp);
#endif
                Instrument::data_access_id_t newDataAccessInstrumentationId;
                DataAccess *insertion = new DataAccess(/*access_type*/READ_ACCESS_TYPE, /*weak*/false, 
                                                       /*originator*/it->_originator, /*range*/tmp, /*fragment*/false, 
                                                       /*instrumentation_id*/newDataAccessInstrumentationId);
                //! Accesses are sorted, so we can directly use push_back.
                collapsedAccesses.push_back(*insertion);
                //std::cerr << "Inserted range [" << collapsedStart << ", " << collapsedEnd << "]." << std::endl;
                collapsedStart = (void *)roundedStart; 
            }
            collapsedEnd = (void *)roundedEnd;
            //! Advance iterator to check if this is the last element
            TaskDataAccesses::accesses_t::iterator prev = it++;
            if(it == accesses._accesses.end()) {
                DataAccessRange tmp = DataAccessRange(collapsedStart, collapsedEnd);
#ifndef NDEBUG
                CacheTrackingSet::checkRange(tmp);
#endif
                Instrument::data_access_id_t newDataAccessInstrumentationId;
                DataAccess *insertion = new DataAccess(/*access_type*/READ_ACCESS_TYPE, /*weak*/false, 
                                                       /*originator*/prev->_originator, /*range*/tmp, /*fragment*/false, 
                                                       /*instrumentation_id*/newDataAccessInstrumentationId);
                collapsedAccesses.push_back(*insertion);
                //std::cerr << "Inserted range [" << collapsedStart << ", " << collapsedEnd << "]." << std::endl;
            }
            return true;
        }
    );
#ifndef NDEBUG
    if(collapsedAccesses.empty())
        assert(accesses._accesses.empty());
    accesses._accesses.processAll(
        [&] (TaskDataAccesses::accesses_t::iterator it) -> bool {
            //std::cerr << "Is range [" << it->getAccessRange().getStartAddress() << ", " << it->getAccessRange().getEndAddress() << "] present?" << std::endl;
            //std::cerr << "Check it: " << std::endl;
            std::size_t size = it->getAccessRange().getSize();
            collapsedAccesses.processIntersecting(
                it->getAccessRange(),
                [&] (TaskDataAccesses::accesses_t::iterator &intersectingKey) -> bool {
                    //std::cerr << "Range [" << intersectingKey->getAccessRange().getStartAddress() << ", " 
                    //          << intersectingKey->getAccessRange().getEndAddress() << "]." << std::endl;
                    size -= it->getAccessRange().intersect(intersectingKey->getAccessRange()).getSize();
                    return true;
                }
            );
            assert(size==0);
            return true;
        }
    );
#endif
    accesses._lock.unlock();
    collapsedAccesses.processAll(
        [&] ( TaskDataAccesses::accesses_t::iterator it ) -> bool {
            //uintptr_t roundedStart = (uintptr_t)it->getAccessRange().getStartAddress();
            //uintptr_t roundedEnd = (uintptr_t)it->getAccessRange().getEndAddress();
            //roundedStart = round_down(roundedStart, cacheLineSize);
            //roundedEnd = round_up(roundedEnd, cacheLineSize);
            //std::size_t actualRangeSize = roundedEnd - roundedStart;
            //DataAccessRange tmp = DataAccessRange((void *)roundedStart, actualRangeSize);
            DataAccessRange tmp = it->getAccessRange();

            //std::cerr << "Processing access with range [" << it->_range.getStartAddress() << ", " << it->_range.getEndAddress() << "]." << std::endl;
            Instrument::Timer timerCacheTrackingSetLock;
            timerCacheTrackingSetLock.start();
            cacheTracking._lock.writeLock();
            timerCacheTrackingSetLock.stop();
            _instance->_timerCacheTrackingSetLock+=timerCacheTrackingSetLock;
            Instrument::Timer timerIntersectingAndMissing;
            timerIntersectingAndMissing.start();
            CacheTrackingSet::iterator previous_insert = cacheTracking.end();
            cacheTracking.processIntersectingAndMissing(
                tmp,
                [&] (CacheTrackingSet::iterator &intersectingKey) -> bool {
                    Instrument::Timer timerIntersecting;
                    timerIntersecting.start();
#ifndef NDEBUG
                    CacheTrackingSet::checkRange(intersectingKey->getAccessRange());
#endif
                    if(intersectingKey->getAccessRange() == tmp) { 
                        //std::cerr << "JUST UPDATE!" << std::endl;
                        Instrument::Timer timerUpdate;
                        timerUpdate.start();
                        cacheTracking.insert(tmp, lastUse, true, intersectingKey);
                        timerUpdate.stop();
                        _instance->_timerUpdate+=timerUpdate;
                    }
                    else if(intersectingKey->getAccessRange().fullyContainedIn(tmp)) {
                        //std::cerr << "ERASE INTERSECTING BECAUSE ITS FULLY CONTAINED IN THE INSERTION." << std::endl;
                        Instrument::Timer timerErase;
                        timerErase.start();
                        cacheTracking.erase(*intersectingKey);
                        cacheTracking.updateCurrentWorkingSetSize(-(intersectingKey->getSize()));
                        timerErase.stop();
                        _instance->_timerErase+=timerErase;
                    }
                    else {
                        //std::cerr << "FRAGMENTING BY INTERSECTION." << std::endl;
                        std::size_t deletedSize = tmp.intersect(intersectingKey->getAccessRange()).getSize();

                        Instrument::Timer timerFragment;
                        timerFragment.start();
                        cacheTracking.fragmentByIntersection(
                            intersectingKey, tmp, true,   
                            [&](CacheTrackingObject const &toBeDuplicated) -> CacheTrackingObject * {
                                CacheTrackingObject * duplicated = new CacheTrackingObject(toBeDuplicated);
#ifndef NDEBUG
                                CacheTrackingSet::checkRange(duplicated->getAccessRange());
#endif
                                return duplicated;
                            },
				            [&](CacheTrackingObject *fragment, CacheTrackingObject *originalCacheTrackingObject) {
                                _unused(fragment);
                                _unused(originalCacheTrackingObject);
#ifndef NDEBUG
                                CacheTrackingSet::checkRange(fragment->getAccessRange());
                                CacheTrackingSet::checkRange(originalCacheTrackingObject->getAccessRange());
#endif
                            }
                        );
                        cacheTracking.updateCurrentWorkingSetSize(-(deletedSize));
                        timerFragment.stop();
                        _instance->_timerFragment+=timerFragment;
                    }

                    timerIntersecting.stop();
                    _instance->_timerIntersecting+=timerIntersecting;
                    return true;
                },
				[&] (DataAccessRange range, CacheTrackingSet::iterator it) -> bool {
                    Instrument::Timer timerMissing;
                    timerMissing.start();
                    //std::cerr << "INSERTING MISSING RANGE [" << range.getStartAddress() << ", " << range.getEndAddress() << "]." << std::endl;
                    uintptr_t missingStart = (uintptr_t)range.getStartAddress();
                    uintptr_t missingEnd = (uintptr_t)range.getEndAddress();
                    missingStart = round_down(missingStart, cacheLineSize);
                    missingEnd = round_up(missingEnd, cacheLineSize);
                    std::size_t actualRangeSize = missingEnd - missingStart;
                    DataAccessRange missing = DataAccessRange((void *)missingStart, actualRangeSize);
					//previous_insert = cacheTracking.insert(missing, lastUse, false, previous_insert);
                    cacheTracking.insert(missing, lastUse, false, it);
                    timerMissing.stop();
                    _instance->_timerMissing+=timerMissing;
					return true;
				}
            );	
            timerIntersectingAndMissing.stop();
            _instance->_timerIntersectingAndMissing+=timerIntersectingAndMissing;

            cacheTracking._lock.writeUnlock();

			return true;
        }
    );
    timerProcess.stop();
    _instance->_timerProcess+=timerProcess;

    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "Listing last level cache tracking set at registerLastLevelCacheData end before evicting." << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //cacheTracking.processAll(
    //    [&] (CacheTrackingSet::iterator it) -> bool {
    //        std::cerr << "Range [" << it->getStartAddress() << ", " << it->getEndAddress() << "] with lastUse " << it->getLastUse() << "." << std::endl;
    //        return true;
    //    }
    //);
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;

    Instrument::Timer timerCacheTrackingSetLock;
    timerCacheTrackingSetLock.start();
    cacheTracking._lock.writeLock();
    timerCacheTrackingSetLock.stop();
    _instance->_timerCacheTrackingSetLock+=timerCacheTrackingSetLock;
    Instrument::Timer timerEvict;
    timerEvict.start();
    cacheTracking.evict();
    timerEvict.stop();
    _instance->_timerEvict+=timerEvict;
    assert(cacheTracking.getCurrentWorkingSetSize() <= cacheTracking.getAvailableLastLevelCacheSize());
    cacheTracking._lock.writeUnlock();

    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "Listing last level cache tracking set at registerLastLevelCacheData end." << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //cacheTracking.processAll(
    //    [&] (CacheTrackingSet::iterator it) -> bool {
    //        std::cerr << "Range [" << it->getStartAddress() << ", " << it->getEndAddress() << "] with lastUse " << it->getLastUse() << "." << std::endl;
    //        return true;
    //    }
    //);
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "Finished registering last level cache data of task " << task->getTaskInfo()->task_label << " (" << task << ") with task data size " 
    //          << task->getDataSize() << ". The number of accesses is " << accesses._accesses.size() <<  "." << std::endl;
    aux.stop();
    _instance->_timer+=aux;
}


double Directory::computeTaskAffinity(Task * task, unsigned int NUMANodeId) {
    _instance->_tasksComputed++;
    assert(_instance->_enableLastLevelCacheTracking);
    assert(task != nullptr && "Task cannot be null");
    if(task->getDataSize() == 0)
        return 0.0;
    Instrument::Timer timerComputeScore;
    timerComputeScore.start();
    double score = 0.0;

    //std::cerr << "Computing score of task " << task->getTaskInfo()->task_label << "." << std::endl;
    CacheTrackingSet &cacheTracking = *(_instance->_lastLevelCacheTracking[NUMANodeId]);
    Instrument::Timer timerTaskDataAccessesLock;
    timerTaskDataAccessesLock.start();
	std::lock_guard<TaskDataAccesses::spinlock_t> guardAccesses(task->getDataAccesses()._lock);
    timerTaskDataAccessesLock.stop();
    _instance->_timerTaskDataAccessesLock+=timerTaskDataAccessesLock;
    //std::lock_guard<SpinLock> guardCache(cacheTracking._lock);
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "Listing last level cache tracking set at computeTaskAffinity start." << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //cacheTracking.processAll(
    //    [&] (CacheTrackingSet::iterator it) -> bool {
    //        std::cerr << "Range [" << it->getStartAddress() << ", " << it->getEndAddress() << "] with lastuse " << it->getLastUse() << "." << std::endl;
    //        return true;
    //    }
    //);
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
	task->getDataAccesses()._accesses.processAll(
		[&] ( TaskDataAccesses::accesses_t::iterator it ) -> bool {
            //std::cerr << "Is this range [" << it->getAccessRange().getStartAddress() << ", " << it->getAccessRange().getEndAddress() << "] in the treap?" << std::endl;

            cacheTracking._lock.readLock();
			cacheTracking.processIntersecting(
				it->getAccessRange(),
				[&] (CacheTrackingSet::iterator &intersectingKey) -> bool {
                    //std::cerr << "IntersectingKey with range [" << intersectingKey->getStartAddress() << ", " << intersectingKey->getEndAddress() << "]." << std::endl;
                    DataAccessRange overlap = it->getAccessRange().intersect(intersectingKey->getAccessRange());
                    assert((overlap.getSize() <= intersectingKey->getSize()) &&
                           (overlap.getSize() <= it->getAccessRange().getSize()));
                    score += overlap.getSize();
					return true;
				}
			);	
            cacheTracking._lock.readUnlock();
            return true;
        }
    );

    //std::cerr << "Task " << task->getTaskInfo()->task_label << " has score before dividing: " << score << "." << std::endl;
    //std::cerr << "Task " << task->getTaskInfo()->task_label << " has data size: " << task->getDataSize() << "." << std::endl;
    score /= task->getDataSize();
    assert(score <= 1.0 && "Affinity score cannot be over 100%.");
    timerComputeScore.stop();
    _instance->_timerComputeScore+=timerComputeScore;
    return score;
}
