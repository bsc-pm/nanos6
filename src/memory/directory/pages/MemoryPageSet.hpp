#ifndef REGION_SET_HPP
#define REGION_SET_HPP

#include <boost/intrusive/avl_set.hpp> //boost::intrusive

#include "MemoryPageObject.hpp"
#include "dependencies/linear-regions/DataAccessRange.hpp"

class MemoryPageSet{

private:
	
	typedef boost::intrusive::member_hook< MemoryPageObject, MemoryPageObject::member_hook_t, &MemoryPageObject::_hook > MemberOption;
	typedef boost::intrusive::avl_multiset< MemoryPageObject, MemberOption, boost::intrusive::key_of_value< MemoryPageObject::key_value > > MemoryPageObjectSet;
	
	MemoryPageObjectSet _set;

public:
	
	typedef MemoryPageObjectSet::iterator iterator;
	typedef MemoryPageObjectSet::const_iterator const_iterator;	

	MemoryPageSet();
	iterator begin();
	iterator end();
	iterator find(void *address);
	iterator insert(DataAccessRange range);	
	
};

#endif //REGION_SET_HPP
