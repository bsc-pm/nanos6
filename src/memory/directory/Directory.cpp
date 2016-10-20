#include "Directory.hpp"

Directory::_instance = nullptr;

Directory::Directory(): _pages(), copies(){}

void Directory::initialize(){
	instance = new Directory();
}

void Directory::dispose(){
	delete _instance;
}

int Directory::copy_version(void *address, size_t size){
	CopySet::iterator it = _instance->copies.find(address, size);
	if(it != _instance->copies.end()){
		return it->getVersion();
	} else {
		return -1;
	}
}

int Directory::insert_copy(void *address, size_t size, GenericCache *cache, bool increment){
	CopySet::iterator it = _instance->_copies.insert(address, size, cache, increment);
	return it->getVersion();
}

void Directory::erase_copy(void *address, GenericCache *cache){
	_instance->_copies.erase(address, GenericCache cache);
}
