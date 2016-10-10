#ifndef MEMORY_OBJECT_HPP
#define MEMORY_OBJECT_HPP

#include <boost/intrusive/avl_set.hpp> //boost::intrusive
#include <functional> //std::less
#include <iostream> // std::ostream (debugging)
#include <numa.h>
#include <numaif.h>

#include "MemoryOperations.hpp"
#include "MemoryRegion.hpp"
#include "CopyObject.hpp"
#include "Shape.hpp"
#include "hardware/Machine.hpp"

class MemoryObject: public boost::intrusive::avl_set_base_hook<> {
	
	typedef boost::intrusive::avl_set< MemoryRegion, boost::intrusive::key_of_value< MemoryRegion::key_value > > RegionSet;
	typedef boost::intrusive::avl_set< CopyObject, boost::intrusive::key_of_value< CopyObject::key_value > > CopySet;

	typedef CopySet::const_iterator copy_iterator;
	typedef RegionSet::const_iterator region_iterator;
private:

	void *_baseAddress; //< base address of the object [_baseAddress, _endAddress)
	void *_endAddress; //< end address of the object [_baseAddress, _endAddress)
	unsigned int _pages; //< number of pages in the object
	Shape _shape; //< shape of the object
	RegionSet _homes; //< list of linear regions in each node
	CopySet _copies; //< list of linear regions replicated in a cache

	

public:

	MemoryObject(void *baseAddress, unsigned int dimensions, unsigned int *shape, std::size_t itemSize)
	: _baseAddress(baseAddress), 
	_shape(dimensions, shape, itemSize){
		_endAddress = MemoryOps::add( _baseAddress, _shape._bytes );
		_pages = MemoryOps::pageNumber(_baseAddress, _endAddress);
	};	
	
	/*
	 * \brief deletes the object and all copies / regions that it contains.
	 *	
	 * Deletes the object and all associated copies and regions. Cannot replace the regular destructor due to the need for reshaping / merging objects, 
	 	where the containers declared in the object are deleted but their elements are copies to another MemoryObject.
	 */
	inline void dispose();

	inline void *getBaseAddress();
	inline void *getEndAddress();
	
	inline bool contains(MemoryObject &a);
	inline bool containedIn(MemoryObject &a);
	inline bool intersects(MemoryObject &a); // x1 <= y2 && y1 <= x2
	
	inline void addCopy(CopyObject &cpy);
	inline void addCopy(std::pair<int, int> *accesses, unsigned int dimensions);

	inline void addRegion(MemoryRegion &reg);
	/*
		NOT YET IMPLEMENTED

	inline void merge(MemoryObject &obj);
	inline void expand(MemoryObject &obj);	
	inline void locate(size_t bytes);
	*/
	
	friend bool operator< (const MemoryObject &a, const MemoryObject &b){  
		return a._baseAddress < b._baseAddress;  
	}
	
	friend bool operator> (const MemoryObject &a, const MemoryObject &b){  
		return a._baseAddress > b._baseAddress;  
	}

	friend bool operator== (const MemoryObject &a, const MemoryObject &b){  
		return a._baseAddress < b._baseAddress;  
	}


	/* Printing in Unit tests */
	friend std::ostream &operator<<(std::ostream &os, const MemoryObject &obj){
		os << "{ MemoryObject: ";
		os << "region [" << obj._baseAddress << "," << obj._endAddress << ")";
		os << " | ";
		os << "shape " << obj._shape << "";
		os << "\nCopies\n";
		for(copy_iterator it = obj._copies.begin(); it != obj._copies.end(); ++it) os <<"\t" << *it << "\n";
		os << " }";
		return os;		
	}

	/* Key for Boost Intrusive AVL Set */
	struct key_value 
	{
		typedef void *type;
		
		const type &operator()(const MemoryObject &m){
			return m._baseAddress;
		}
	};

	friend key_value;
};

/* Implementation */

struct delete_copies
{
   void operator()(CopyObject *delete_this)
   {  delete delete_this;  }
};

struct delete_regions
{
   void operator()(MemoryRegion *delete_this)
   {  delete delete_this;  }
};
	
struct move_regions{
	MemoryObject *obj;
	
	move_regions(MemoryObject *obj): obj(obj){}
	
	void operator()(MemoryRegion *move_this){ 
		obj->addRegion(*move_this); 
	}
	
};

struct move_copies{
	MemoryObject *obj;
	
	move_copies(MemoryObject *obj): obj(obj){}
	
	void operator()(CopyObject *move_this){ 
		obj->addCopy(*move_this); 
	}

};

typedef struct delete_copies delete_copies;
typedef struct delete_regions delete_regions;
typedef struct move_copies move_copies;
typedef struct move_regions move_regions;

/* END STRUCTS */

void MemoryObject::dispose(){
	_homes.clear_and_dispose( delete_regions() );
	_copies.clear_and_dispose( delete_copies() );

	delete this;
}

void *MemoryObject::getBaseAddress(){
	return _baseAddress;
}

void *MemoryObject::getEndAddress(){
	return _endAddress;
}

bool MemoryObject::contains(MemoryObject &a){	
	return ( _baseAddress <= a._baseAddress && _endAddress >= a._endAddress ); 
}

bool MemoryObject::containedIn(MemoryObject &a){ 
	return ( _baseAddress >= a._baseAddress && _endAddress <= a._endAddress ); 
}

bool MemoryObject::intersects(MemoryObject &a){ 
	return ( _baseAddress <= a._endAddress && a._baseAddress <= _endAddress ); 
}

void MemoryObject::addCopy(CopyObject &copy){
	_copies.insert(copy);
}

void MemoryObject::addCopy(std::pair<int, int> *accesses, unsigned int dimensions){
	CopyObject *copy = new CopyObject(this, dimensions, accesses, 0 /* Need to calculate version somehow */);
	_copies.insert(*copy);
}

void MemoryObject::addRegion(MemoryRegion &reg){
	_homes.insert(reg);
} 

/*   IMPLEMENTED, NOT YET NEEDED
//Use clear_and_dispose

void MemoryObject::merge(MemoryObject &other){
	//Merge the home and cache trees
	other._homes.clear_and_dispose( MemoryObject::move_homes( this ) );	
	other._cache.clear_and_dispose(	MemoryObject::move_cache( this ) );
}

void MemoryObject::expand(MemoryObject &other){
	std::size_t bytes = 0;
	if( _baseAddress > other._baseAddress ) bytes += MemoryOps::distance(_baseAddress, other._baseAddress);
	if( other._endAddress > _endAddress ) bytes += MemoryOps::distance(_endAddress, other._endAddress); 
		
	_shape.reshape(bytes);
}

void MemoryObject::locate(size_t bytes){
	long pagesize = Machine::instance()->getPageSize();
	
	int npages = bytes / pagesize; // TODO This needs to be a ceil, not sure if floor
	void **pages = new void*[npages];
	int *status = new int[npages];	
	
	// Fill pages with the starting addresses of the pages on the object
	pages[0] = (void*)( ( long )_baseAddress & ~(pagesize-1) );
	for(int i = 1; i < npages; i++){
		pages[i] = pages[0] + i * pagesize;
	}

	move_pages(0, npages, pages, NULL, status, 0);
	
	//Create the memory regions
	
	int node = -1; //Node number of the region being allocated
	void *base = pages[0];
	int size = 0;
	for(int i = 0; i < npages; i++){
		if(status[i] != node){
			if(size != 0){
				MemoryRegion *m = new MemoryRegion(base, size); //TODO add node (reference, pointer or number)
				_homes.insert(*m);

				size = 0;
				base = pages[i];
			}
			node = status[i];
		}	
		size += pagesize;
	}	
}

*/



#endif //MEMORY_OBJECT_HPP
