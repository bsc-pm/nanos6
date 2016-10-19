#ifndef DIRECTORY_HPP
#define DIRECTORY_HPP

#include <boost/intrusive/avl_set.hpp> //boost::intrusive
#include <functional> // std::less

#include "CopySet.hpp"
#include "RegionSet.hpp"

#include "memory/TaskMemoryData.hpp"

class Directory {


private:
	CopySet _copies;
	RegionSet _regions;	

public:	
	//typedef ObjectSet::iterator iterator;
	
	Directory()
	: _copies(),
	_regions(){		
	}

	/* At least base address and shape needed */	

	void insert(TaskMemoryData data);
	
	void /* std::pair<iterator, iterator>? */ erase(/* Define */);
};

#endif //DIRECTORY_HPP
