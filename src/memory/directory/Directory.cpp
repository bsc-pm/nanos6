#include "Directory.hpp"

int Directory::insert_copy(void *baseAddress, size_t size, GenericCache *cache, bool increment){
	CopySet::iterator it = _copies.insert(baseAddress, size, cache, increment);
	return it->getVersion();
}

void Directory::erase_copy(void *baseAddress, GenericCache *cache){
	_copies.erase(baseAddress, GenericCache cache);
}
