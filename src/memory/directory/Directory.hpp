#ifndef DIRECTORY_HPP
#define DIRECTORY_HPP

#include <boost/intrusive/avl_set.hpp> //boost::intrusive
#include <functional> // std::less

#include <TaskDataAccesses.hpp>

#include "copies/CopySet.hpp"
#include "pages/MemoryPageSet.hpp"

class Directory {


private:
	CopySet _copies;
	MemoryPageSet _pages;	

	static Directory *_instance;

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
	 *
	 *	\param address The starting address of the copy
	 *	\param size The size of the copy
	 */
	static int copy_version(void *address);

	/*! \brief Registers a region that has been copied at a certain cache 
	 * 
	 *  \param address The base address of the copied region
	 *  \param size The size of the copied region
	 *  \param cache The cache to which the region is copied
	 *  \param increment True if the version needs to be incremented
	 */
	static int insert_copy(void *address, size_t size, int cache, bool increment);

	/*! \brief Registers a region that has been removed from a cache
	 *  
	 *  \param address The base address of the evicted region
	 *	\param size Size of the evicted region
	 *	\param cache The cache from which the region is evicted
	 */
	static void erase_copy(void *address, int cache);

	/*! \brief Retrieves location data of the data accesses of a task in order to determine execution place.
	 *	
	 *  Accesses the information in the directory in order to determine in which processors / node will a task be executed.
	 *	First it accesses the home node information of each access and registers it inside the access. 
	 *	If the region is not registered move_pages will be called to retrieve the information.
	 *	Second it checks if any of the regions has a copy present in any cache and copies the information to the access.	
	 *	
	 *	How the policy is involved in this process is still to be discussed
	 *
	 *	\param accesses Data accesses of a Task
	 *
	 */
	static void /*provisional name*/analyze(TaskDataAccesses &accesses /* Policy here? */);	
};

#endif //DIRECTORY_HPP
