#ifndef MEMORY_OBJECT_CPP
#define MEMORY_OBJECT_CPP

#include <boost/intrusive/avl_set.hpp> //boost::intrusive
#include <functional> //std::less
#include <iostream> // std::ostream (debugging)

#include "MemoryRegion.hpp"
#include "CacheObject.hpp"

class MemoryObject: public boost::intrusive::avl_set_base_hook<> {
	
	typedef boost::intrusive::avl_set< MemoryRegion, boost::intrusive::key_of_value< MemoryRegion::key_value > > RegionSet;
	typedef boost::intrusive::avl_set< CacheObject, boost::intrusive::key_of_value< CacheObject::key_value > > CacheSet;

private:
	void * _baseAddress; //< base address of the object
	// SHAPE
	RegionSet _homes; //< list of linear regions in each node
	CacheSet _cached; //< list of liner regions replicated in a cache

public:
	MemoryObject(void *baseAddress)
	: _baseAddress(baseAddress),
	_homes(),
	_cached(){

	}

	void *getBaseAddress(){
		return _baseAddress;
	}

	/* Comparison operators for Boost Intrusive AVL Set (OLD) */
	friend bool operator <(const MemoryObject &a, const MemoryObject &b){ return a._baseAddress < b._baseAddress; }
	friend bool operator >(const MemoryObject &a, const MemoryObject &b){ return a._baseAddress > b._baseAddress; }
	friend bool operator ==(const MemoryObject &a, const MemoryObject &b){ return a._baseAddress == b._baseAddress; }
	
	/* Debbuging / Testing */
	friend std::ostream &operator<<(std::ostream &os, const MemoryObject &obj){
		os << "Base Address: " << obj._baseAddress << std::endl;
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

#endif //MEMORY_OBJECT_CPP
