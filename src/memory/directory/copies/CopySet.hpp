#ifndef COPY_SET_HPP
#define COPY_SET_HPP

#include <boost/intrusive/avl_set.hpp> //boost::intrusive
#include <functional> // std::less

#include <IntrusiveLinearRegionMap.hpp>
#include <IntrusiveLinearRegionMapImplementation.hpp>

#include "CopyObject.hpp"

class CopySet: public IntrusiveLinearRegionMap<CopyObject, boost::intrusive::function_hook< CopyObjectLinkingArtifacts > >{

	 

private:
	typedef IntrusiveLinearRegionMap<CopyObject, boost::intrusive::function_hook< CopyObjectLinkingArtifacts > > BaseType;

public:

	CopySet();

	iterator find(void *start, void *end);
	iterator find(void *address);
	
	/*! \brief Inserts a copy in a cache to the list 
	 *
	 * 	Registers that a region has been copied in a cache.
	 * 	If the region is already present in the list it adds a new one (if not already present) to the list of copies of the region.
	 * 	If the region was not on the list it creates a new CopyObject and adds the cache to its list.
	 * 	Optionally increments the version of the copy.
	 *
	 *  Returns the highest version.  
     *
	 *	\param address Starting address of the copy region
	 *	\param size Size of the copy region
     *  \param homeNode the homeNode of the copy region 
	 *	\param cache Cache object where the copy is stored
	 *	\param increment True if the version must be incremented
	 */
	int insert(void *address, size_t size, int homeNode, int cache, bool increment);
	
	/*!	\brief Removes a copy on a cache from the list
	 *
	 *	Registers that a region has been evicted from a cache.
	 * 	If the CopyObject had more up-to-date copies, it will remove the cache from the list.
	 *	If the CopyObject only had this copy it will remove the object from the directory. 
	 *
	 *	\param address Starting address of the copy region
	 *	\param cache Cache from which the copy was evicted
	 */
	iterator erase(void *address, int cache);
};

#endif //COPY_SET_HPP
