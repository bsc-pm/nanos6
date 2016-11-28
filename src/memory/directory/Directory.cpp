#include <mutex>

#include "Directory.hpp"
#include <TaskDataAccessesImplementation.hpp>

Directory *Directory::_instance = nullptr;

Directory::Directory(): _pages(), _copies(), _lock(){}

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

void Directory::analyze(TaskDataAccesses &accesses, size_t *vector){
	std::lock_guard<SpinLock> guard(accesses._lock);
	
	// Process all Data accesses
	accesses._accesses.processAll(
		[&] ( TaskDataAccesses::accesses_t::iterator it ) -> bool {

			// Process all possible gaps in the pages directory and insert them in the pages list
			Directory::_instance->_pages.processMissing(
				it->getAccessRange(),
				[&] (DataAccessRange missingRange) -> bool {
					Directory::_instance->_pages.insert(missingRange);
					return true;
				}
			);	

			// Search for all pages in the pages list
			Directory::_instance->_pages.processIntersecting(
				it->getAccessRange(),
				[&] (MemoryPageSet::iterator position) -> bool {
					it->_homeNode = position->getLocation();
					return true; 
				}
			);
		
			// Process the intersecting copies for each access
			Directory::_instance->_copies.processIntersecting(
				it->getAccessRange(),
				
				// Process regions which are present in the directory
				// The size of the copyObjects is added to the positions corresponding to the caches in which they are
				[&] (CopySet::iterator position ) -> bool {
					
					size_t size = position->getSize();
						
					// Is the current copy is an edge, substract the outer part from the size
					if(position->getStartAddress() < it->getAccessRange().getStartAddress()){
						size -= ( static_cast<char *>( it->getAccessRange().getStartAddress() ) - static_cast<char *>( position->getStartAddress() ) );
					}
					if( position->getEndAddress() > it->getAccessRange().getEndAddress() ){
						size -= ( static_cast<char *>( position->getEndAddress() ) - static_cast<char *>( it->getAccessRange().getEndAddress() ) );
					}					
					
					// Add the size to the corresponding position in the vector.
					for(int i = 0; i < position->countCaches(); i++){							
						if( position->testCache(i) || ((position->getHomeNode() == i) && (position->isHomeNodeUpToDate()))){
							vector[i] += size;
						}
					} 
					
					return true;	
				}
			);				
			return true;
		} 
	);	
}

cache_mask Directory::getCaches(void *address) {
	CopySet::iterator it = Directory::_instance->_copies.find(address);
    assert(it != _instance->_copies.end() && "The copy must be in the directory");
    return it->getCaches();
}

int Directory::getHomeNode(void *address) {
    CopySet::iterator it = _instance->_copies.find(address);
    if(it != _instance->_copies.end()) {
        return it->getHomeNode();
    }
    else {
        return -1;
    }
}

bool Directory::isHomeNodeUpToDate(void *address) {
    CopySet::iterator it = _instance->_copies.find(address);
    //assert(it != _instance->_copies.end() && "The copy must be in the directory");
    if(it != _instance->_copies.end()) {
        return it->isHomeNodeUpToDate();
    }
    else {
        return false;
    }
}

void Directory::setHomeNodeUpToDate(void *address, bool b) {
    CopySet::iterator it = _instance->_copies.find(address);
    //assert(it != _instance->_copies.end() && "The copy must be in the directory");
    if(it != _instance->_copies.end()) {
        it->setHomeNodeUpToDate(b);
    }
}
