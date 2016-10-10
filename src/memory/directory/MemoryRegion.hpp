#ifndef REGION_HPP
#define REGION_HPP

#include <boost/intrusive/avl_set.hpp>
#include "hardware/places/MemoryPlace.hpp"
#include "memory/Region.hpp"

class MemoryRegion : public boost::intrusive::avl_set_base_hook<> {
	
private:
	Region _region;
	MemoryPlace *_location; //< memory nodes of the region

public:

	MemoryRegion( void *baseAddress, size_t size, MemoryPlace *location = nullptr )
		: _region(baseAddress, size),
		_location( location )
	{
	
	}

	
	/* Key structure for Boost Intrusive AVL Set */
	struct key_value
	{
		typedef void *type;
		
		const type &operator()(const MemoryRegion &m){
			return m._region._baseAddress;
		}
	};
	
	friend key_value;
};

#endif //REGION_HPP
