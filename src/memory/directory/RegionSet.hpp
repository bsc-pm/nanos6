#ifndef REGION_SET_HPP
#define REGION_SET_HPP

#include <boost/intrusive/avl_set.hpp> //boost::intrusive
#include <functional> // std::less

#include "MemoryRegion.hpp"
#include "memory/TaskMemoryData.hpp"

class RegionSet{

private:

	typedef boost::intrusive::avl_set< MemoryRegion, boost::intrusive::key_of_value< MemoryRegion::key_value > > Set;
	
	Set _set;

public:
	
	typedef Set::iterator iterator;
	typedef Set::const_iterator const_iterator;	

	RegionSet(): _set(){

	};

	iterator begin();
	iterator end();
	iterator find(void *address);
	vector<MemoryRegion> insert(TaskMemoryData data);

};

#endif //REGION_SET_HPP
