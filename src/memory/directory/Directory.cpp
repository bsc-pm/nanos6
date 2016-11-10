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

int Directory::copy_version(void *address){
	CopySet::iterator it = Directory::_instance->_copies.find(address);
	if(it != Directory::_instance->_copies.end()){
		return it->getVersion();
	} else {
		return -1;
	}

}

int Directory::insert_copy(void *address, size_t size, int cache, bool increment){
	std::lock_guard<SpinLock> guard(Directory::_instance->_lock);
	return Directory::_instance->_copies.insert(address, size, cache, increment);
}

void Directory::erase_copy(void *address, int cache){
	std::lock_guard<SpinLock> guard(Directory::_instance->_lock);
	Directory::_instance->_copies.erase(address, cache);
}

void Directory::analyze(TaskDataAccesses &accesses, size_t *vector){
	std::lock_guard<SpinLock> guard(accesses._lock);
	
	// Process all Data accesses
	accesses._accesses.processAll(
		[&] ( TaskDataAccesses::accesses_t::iterator it ) -> bool {
			// Process the intersecting copies (and spaces in between) for each access
			Directory::_instance->_copies.processIntersectingAndMissing(
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
						if( position->testCache(i) ){
							vector[i] += size;
						}
					} 
					
					return true;	
				},

				// Process regions which are missing from the directory
				
				[&] (DataAccessRange missingRange) -> bool {
					// Do nothing for now, potentially use the homes information
					return true;
				}
			);				
			return true;
		} 
	);	
	// 
}
