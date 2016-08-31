#ifndef REGION_HPP
#define REGION_HPP

#include <boost/intrusive/avl_set.hpp>
#include "hardware/places/MemoryPlace.hpp"

class MemoryRegion : public boost::intrusive::avl_set_base_hook<> {
	
private:
	void *_baseAddress; //< start address of the region
	int _size; //< length of the region
	MemoryPlace *_location; //< memory nodes of the region

public:

	MemoryRegion( void *baseAddress, size_t size, MemoryPlace *location = nullptr )
		: _baseAddress( baseAddress ),
		_size( size ),
		_location( location )
	{
	
	}

	
	/* Comparison operators for Boost Intrusive AVL Set (OLD) */ 
	friend bool operator <(const MemoryRegion &a, const MemoryRegion &b) { return a._baseAddress < b._baseAddress; }
	friend bool operator >(const MemoryRegion &a, const MemoryRegion &b) { return a._baseAddress > b._baseAddress; }
	friend bool operator ==(const MemoryRegion &a, const MemoryRegion &b) { return a._baseAddress == b._baseAddress; } 

	/* Debugging / Testing */
	friend std::ostream& operator<<(std::ostream &os, const MemoryRegion &obj){
        void *end = static_cast<char*>(obj._baseAddress) + obj._size;
        os << "{ MemoryRegion: ";
		
		os << "region [" << obj._baseAddress << "-" << end <<")";

		os << " | ";
		
		os << "location: " << 0;
		
		os << " }";

        return os;
    }


	/* Key structure for Boost Intrusive AVL Set */
	struct key_value
	{
		typedef void *type;
		
		const type &operator()(const MemoryRegion &m){
			return m._baseAddress;
		}
	};
	
	friend key_value;
};

#endif //REGION_HPP
