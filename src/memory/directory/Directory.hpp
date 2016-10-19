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

	/*! Registers a region that has been copied at a certain cache 
	 * 
	 *  \param baseAddress The base address of the copied region
	 *  \param size The size of the copied region
	 *  \param cache The cache to which the region is copied
	 *  \param increment True if the version needs to be incremented
	 */
	int insert_copy(void *baseAddress, size_t size, GenericCache *cache, bool increment);

	/*! Registers a region that has been removed from a cache
	 *  
	 *  \param baseAddress The base address of the evicted region
	 *	\param cache The cache from which the region is evicted
	 */
	void erase_copy(void *baseAddress, GenericCache *cache);
};

#endif //DIRECTORY_HPP
