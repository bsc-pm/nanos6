#ifndef COPY_SET_HPP
#define COPY_SET_HPP

#include <boost/intrusive/avl_set.hpp> //boost::intrusive
#include <functional> // std::less

#include "CopyObject.hpp"

class CopySet {

typedef boost::intrusive::member_hook< CopyObject, CopyObject::member_hook_t, &CopyObject::_hook > MemberOption;
typedef boost::intrusive::avl_multiset< CopyObject, MemberOption, boost::intrusive::key_of_value< CopyObject::key_value > > CopyObjectSet;

private:
	CopyObjectSet _set;

public:
	typedef CopyObjectSet::iterator iterator;
	typedef CopyObjectSet::const_iterator const_iterator;	

	CopySet();
	
	/*! \brief Proxy call for the boost intrusive avl_set begin method*/
	iterator begin();
	
	/*! \brief Proxy call for the boost intrusive avl_set end method*/
	iterator end();
	
	/*! \brief Proxy call for the boost intrusive avl_set find method*/
	iterator find(void *address);

	/*!	\brief Find the CopyObject with the specified address and size
	 *	
	 *	\param address Starting address of the copy region
	 */	
	iterator find(void *address, size_t size);

	/*! \brief Inserts a copy in a cache to the list 
	 *
	 * 	Registers that a region has been copied in a cache.
	 * 	If the region is already present in the list it adds a new one (if not already present) to the list of copies of the region.
	 * 	If the region was not on the list it creates a new CopyObject and adds the cache to its list.
	 * 	Optionally increments the version of the copy.
	 *
	 *	\param address Starting address of the copy region
	 *	\param size Size of the copy region
	 *	\param cache Cache object where the copy is stored
	 *	\param increment True if the version must be incremented
	 */
	iterator insert(void *address, size_t size, int cache, bool increment);
	
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
