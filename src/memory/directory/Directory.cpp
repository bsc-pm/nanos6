#include "Directory.hpp"

Directory *Directory::_instance = nullptr;

Directory::Directory(): _pages(), _copies(){}

void Directory::initialize(){
	Directory::_instance = new Directory();
}

void Directory::dispose(){
	delete Directory::_instance;
}

int Directory::copy_version(void *address){
	//CopySet::iterator it = Directory::_instance->_copies.find(address);
	//if(it != Directory::_instance->_copies.end()){
	//	return it->getVersion();
	//} else {
		return -1;
	//}

}

int Directory::insert_copy(void *address, size_t size, int cache, bool increment){
	//CopySet::iterator it = Directory::_instance->_copies.insert(address, size, cache, increment);
	//return it->getVersion();
	return 0; // IMPLEMENTING
}

void Directory::erase_copy(void *address, int cache){
	//Directory::_instance->_copies.erase(address, cache);
}

void Directory::analyze(TaskDataAccesses &accesses){
	//For loop, retrieve node information of each access

	// 
}
