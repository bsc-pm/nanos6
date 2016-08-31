#ifndef DIRECTORY_HPP
#define DIRECTORY_HPP

#include <boost/intrusive/avl_set.hpp> //boost::intrusive
#include <functional> // std::less

#include "MemoryObject.hpp"

class Directory {

typedef boost::intrusive::avl_set< MemoryObject, boost::intrusive::key_of_value< MemoryObject::key_value > > ObjectSet;

private:
	ObjectSet _set;

public:	
	typedef ObjectSet::iterator iterator;
	
	Directory()
	: _set(){
		
	}

	bool empty();
	int size();
	iterator begin();
	iterator end();
	iterator find(void *address);
	void insert(void* baseAddress, int dimensions, int *dimSizes);
	iterator erase(iterator position);	
};

#endif //DIRECTORY_HPP
