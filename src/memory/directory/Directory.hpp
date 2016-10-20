#ifndef DIRECTORY_HPP
#define DIRECTORY_HPP

#include <boost/intrusive/avl_set.hpp> //boost::intrusive
#include <functional> // std::less

#include "CopySet.hpp"
#include "MemoryPageSet.hpp"

class Directory {


private:
	CopySet _copies;
	MemoryPageSet _pages;	

	static Directory _instance;

	Directory();
public:	
	
	/*! Initializes the directory */
	static void initialize();
	
	/*! Delete method for the directory */
	static void dispose();

	/*!	\brief Returns the version of a copy or -1 if it is not present 
	 *	
	 *	Return the last version of a copy object.
	 *	If the copy is not on the directory a -1 is returned instead.
	 *	The size of the region is required since multiple regions with the same start and different size may exist.
	 *
	 *	\param address The starting address of the copy
	 *	\param size The size of the copy
	 */
	static int copy_version(void *address, size_t size);

	/*! Registers a region that has been copied at a certain cache 
	 * 
	 *  \param address The base address of the copied region
	 *  \param size The size of the copied region
	 *  \param cache The cache to which the region is copied
	 *  \param increment True if the version needs to be incremented
	 */
	static int insert_copy(void *address, size_t size, GenericCache *cache, bool increment);

	/*! Registers a region that has been removed from a cache
	 *  
	 *  \param address The base address of the evicted region
	 *	\param cache The cache from which the region is evicted
	 */
	static void erase_copy(void *address, GenericCache *cache);	
};

#endif //DIRECTORY_HPP
