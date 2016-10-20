#ifndef REGION_HPP
#define REGION_HPP

#include <DataAccessRange.h>

#include <boost/intrusive/avl_set.hpp>
#include "hardware/places/MemoryPlace.hpp"

class MemoryPageObject{
	
private:
	DataAccessRange _range; //< Range of the page
	MemoryPlace *_location; //< memory node where the page resides

public:
 	#if NDEBUG
		typedef boost::intrusive::avl_set_member_hook<boost::intrusive::link_mode<boost::intrusive::normal_link>> member_hook_t;
	#else
		typedef boost::intrusive::avl_set_member_hook<boost::intrusive::link_mode<boost::intrusive::safe_link>> member_hook_t;
	#endif	

	member_hook_t _hook;


	MemoryRegion( void *baseAddress, size_t size, MemoryPlace *location = nullptr );
	void *getStartAddress();
	size_t getSize();
	MemoryPlace *getLocation();

	
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
