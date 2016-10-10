#ifndef COPY_SET_HPP
#define COPY_SET_HPP

#include <boost/intrusive/avl_set.hpp> //boost::intrusive
#include <functional> // std::less

#include "CopyObject.hpp"

class CopySet {

typedef boost::intrusive::avl_multiset< CopyObject, boost::intrusive::key_of_value< CopyObject::key_value > > Set;


private:
	Set _set;

public:
	typedef Set::iterator iterator;
	typedef Set::const_iterator const_iterator;	

	CopySet(): _set(){

	}
	
	iterator begin();
	iterator end();
	iterator find(void *address);

};

#endif //COPY_SET_HPP
