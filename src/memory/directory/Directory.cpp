#include <mutex>

#include "Directory.hpp"
#include <TaskDataAccessesImplementation.hpp>
#include <iostream>

#define _unused(x) ((void)(x))

Directory *Directory::_instance = nullptr;

Directory::Directory(): _lock(), _copies(), _pages(){}

void Directory::initialize(){
	Directory::_instance = new Directory();
}

void Directory::dispose(){
	delete Directory::_instance;
}

int Directory::getVersion(void *address){
	CopySet::iterator it = Directory::_instance->_copies.find(address);
	if(it != Directory::_instance->_copies.end()){
		return it->getVersion();
	} else {
		return -1;
	}
}

int Directory::insertCopy(void *address, size_t size, int homeNode, int cache, bool increment){
	std::lock_guard<SpinLock> guard(Directory::_instance->_lock);
	return Directory::_instance->_copies.insert(address, size, homeNode, cache, increment);
}

void Directory::eraseCopy(void *address, int cache){
	std::lock_guard<SpinLock> guard(Directory::_instance->_lock);
	Directory::_instance->_copies.erase(address, cache);
}

std::vector<double>  Directory::computeNUMANodeAffinity(TaskDataAccesses &accesses ){
	std::lock_guard<SpinLock> guardAccesses(accesses._lock);
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

			//! Process the intersecting copies for each access
            /** For the time being, disable this part **/
			//Directory::_instance->_copies.processIntersecting(
			//	it->getAccessRange(),
			//	
			//	//! Process regions which are present in the directory
			//	//! The size of the copyObjects is added to the positions corresponding to the caches in which they are
			//	[&] (CopySet::iterator position ) -> bool {
			//		
			//		size_t size = position->getSize();
			//			
			//		// Is the current copy is an edge, substract the outer part from the size
			//		if(position->getStartAddress() < it->getAccessRange().getStartAddress()){
			//			size -= ( static_cast<char *>( it->getAccessRange().getStartAddress() ) - static_cast<char *>( position->getStartAddress() ) );
			//		}
			//		if( position->getEndAddress() > it->getAccessRange().getEndAddress() ){
			//			size -= ( static_cast<char *>( position->getEndAddress() ) - static_cast<char *>( it->getAccessRange().getEndAddress() ) );
			//		}					
			//		
			//		// Add the size to the corresponding position in the vector.
			//		for(int i = 0; i < position->countCaches(); i++){							
			//			if( position->testCache(i) || ((position->getHomeNode() == i) && (position->isHomeNodeUpToDate()))){
			//				vector[i] += size;
			//			}
			//		} 
			//		
			//		return true;	
			//	}
			//);				
			return true;
		} 
	);	
    return result;
}

cache_mask Directory::getCaches(void *address) {
	std::lock_guard<SpinLock> guard(Directory::_instance->_lock);
	CopySet::iterator it = Directory::_instance->_copies.find(address);
    assert(it != _instance->_copies.end() && "The copy must be in the directory");
    return it->getCaches();
}

int Directory::getHomeNode(void *address) {
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
	std::lock_guard<SpinLock> guard(Directory::_instance->_lock);
    CopySet::iterator it = _instance->_copies.find(address);
    //assert(it != _instance->_copies.end() && "The copy must be in the directory");
    if(it != _instance->_copies.end()) {
        it->setHomeNodeUpToDate(b);
    }
}

void Directory::addLastLevelCacheTrackingNode(unsigned int NUMANodeId) {
    CacheTrackingSet *a = new CacheTrackingSet();
    _instance->_lastLevelCacheTracking.insert(std::make_pair(NUMANodeId, a));
}

void Directory::registerLastLevelCacheData(TaskDataAccesses &accesses, unsigned int NUMANodeId, Task * task) {
    _unused(task);
    CacheTrackingSet &cacheTracking = *(_instance->_lastLevelCacheTracking[NUMANodeId]);
    long unsigned int lastUse = cacheTracking.getLastUse();
	std::lock_guard<SpinLock> guardAccesses(accesses._lock);
    std::lock_guard<SpinLock> guardCache(cacheTracking._lock);
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "Registering last level cache data of task " << task->getTaskInfo()->task_label << " with task data size " 
    //          << task->getDataSize() << ". The number of accesses is " << accesses._accesses.size() <<  "." << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    accesses._accesses.processAll(
        [&] ( TaskDataAccesses::accesses_t::iterator it ) -> bool {

            //std::cerr << "Processing access with range [" << it->_range.getStartAddress() << ", " << it->_range.getEndAddress() << "]." << std::endl;
            cacheTracking.processIntersecting(
                it->getAccessRange(),
                [&] (CacheTrackingSet::iterator &intersectingKey) -> bool {
                    _unused(intersectingKey);
                    //! Update lastUse of already present accesses to prevent them to be evicted. 
                    //std::cerr << "Updating range" << std::endl;
                    //cacheTracking.insert(intersectingKey->getAccessRange(), lastUse); 
                    cacheTracking.insert(it->getAccessRange(), lastUse);
                    //assert((it->getAccessRange().getStartAddress() == intersectingKey->getStartAddress()) && 
                    //       (it->getAccessRange().getSize() == intersectingKey->getSize()) && 
                    //       "Can't do only update because it's not the same access.");
                    return true;
                }
            );	

			cacheTracking.processMissing(
				it->getAccessRange(),
				[&] (DataAccessRange range) -> bool {
                    //std::cerr << "Inserting range" << std::endl;
					cacheTracking.insert(range, lastUse);
					return true;
				}
			);	
			return true;
        }
    );

    //std::cerr << "----------------------------------------------------------------------------------------------------" << std::endl;
    //std::cerr << "Listing last level cache tracking set." << std::endl;
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
}


double Directory::computeTaskAffinity(Task * task, unsigned int NUMANodeId) {
    assert(task != nullptr && "Task cannot be null");
    if(task->getDataSize() == 0)
        return 0.0;
    double score = 0.0;

    CacheTrackingSet &cacheTracking = *(_instance->_lastLevelCacheTracking[NUMANodeId]);
	std::lock_guard<SpinLock> guardAccesses(task->getDataAccesses()._lock);
    std::lock_guard<SpinLock> guardCache(cacheTracking._lock);
	task->getDataAccesses()._accesses.processAll(
		[&] ( TaskDataAccesses::accesses_t::iterator it ) -> bool {

			cacheTracking.processIntersecting(
				it->getAccessRange(),
				[&] (CacheTrackingSet::iterator &intersectingKey) -> bool {
                    _unused(intersectingKey);
                    score += it->getAccessRange().getSize();
					return true;
				}
			);	
            return true;
        }
    );

    std::cerr << "Task " << task->getTaskInfo()->task_label << " has score before dividing: " << score << "." << std::endl;
    std::cerr << "Task " << task->getTaskInfo()->task_label << " has data size: " << task->getDataSize() << "." << std::endl;
    score /= task->getDataSize();
    assert(score <= 1.0 && "Affinity score cannot be over 100%.");
    return score;
}
