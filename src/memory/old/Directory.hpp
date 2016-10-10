#ifndef DIRECTORY_HPP
#define DIRECTORY_HPP

#include <boost/intrusive/avl_set.hpp> //boost::intrusive
#include <functional> // std::less

#include "MemoryObject.hpp"

class Directory {

typedef boost::intrusive::avl_multiset< MemoryObject, boost::intrusive::key_of_value< MemoryObject::key_value > > ObjectSet;

private:
	ObjectSet _set;

public:	
	typedef ObjectSet::iterator iterator;
	
	Directory()
	: _set(){
		
	}

	inline bool empty();
	inline int size();
	inline iterator begin();
	inline iterator end();
	inline iterator lower_bound(void *address);
	inline iterator find(void *address);
	inline void insert(void* baseAddress, unsigned int dimensions, unsigned int *shape, std::size_t itemSize);
	inline iterator erase(iterator position);	
};

/* Implementation */

static void *getEndAddress(void* baseAddress, int dimensions, size_t *dimSizes){
	int n_bytes = dimSizes[dimensions-1];
	for(int i = dimensions-2; i >= 0; i--){
		n_bytes = dimSizes[i] * n_bytes;
	}
	return static_cast<void *>( static_cast<char *>( baseAddress ) + n_bytes );
}

bool Directory::empty(){
	return _set.empty();
}

int Directory::size(){
	return _set.size();
}

typename Directory::iterator Directory::begin(){
	return _set.begin(); 
}

typename Directory::iterator Directory::end(){
	return _set.end();
}

typename Directory::iterator Directory::lower_bound(void *address){
	return _set.lower_bound(address);
}

typename Directory::iterator Directory::find(void *address){
	return _set.find(address);
}

void Directory::insert(void *baseAddress, unsigned int dimensions, unsigned int *shape, std::size_t itemSize){
	/*if(!_set.empty()){
		
		MemoryObject *obj = new MemoryObject(baseAddress, dimensions, shape); //penalty for deletion (other option is using a vector to allocate memoryObjects, also penalty)
		Directory::iterator it = _set.lower_bound(baseAddress);	
		
		if(it != _set.begin() && ( it != _set.end()  ||  baseAddress < it->getBaseAddress() ) ){
			it--;
		}	
	
		for(it; it != _set.upper_bound(endAddress) || it != _set.end(); ++it){
			if(it->contains(baseAddress, endAddress) ){
				delete obj;
				return; //contained in something no need to continue
			} else if( it->containedIn(baseAddress, endAddress) || it->intersects(baseAddress, endAddress) ){
			
			}
		}
	}*/	
	

	MemoryObject *obj = new MemoryObject(baseAddress, dimensions, shape, itemSize);
	_set.insert( *obj );
	

}

typename Directory::iterator Directory::erase(typename Directory::iterator position){
	MemoryObject* obj = &(*position);
	Directory::iterator it = _set.erase(position);
	delete obj;
	return it;
}

#endif //DIRECTORY_HPP
