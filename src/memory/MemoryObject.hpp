#ifndef MEMORY_OBJECT_CPP
#define MEMORY_OBJECT_CPP

#include <boost/intrusive/avl_set.hpp> //boost::intrusive
#include <functional> //std::less
#include <iostream> // std::ostream (debugging)
#include <climits> // CHAR_BIT

#include "MemoryRegion.hpp"
#include "CacheObject.hpp"

class MemoryObject: public boost::intrusive::avl_set_base_hook<> {
	
	typedef boost::intrusive::avl_set< MemoryRegion, boost::intrusive::key_of_value< MemoryRegion::key_value > > RegionSet;
	typedef boost::intrusive::avl_set< CacheObject, boost::intrusive::key_of_value< CacheObject::key_value > > CacheSet;

private:

	/* Shape of the memory object (Dimensions and Number of elements in each */
	typedef struct Shape
	{
		friend class MemoryObject;
		
		int _bits; //< number of bits of the shape
		int _bytes; //< number of bytes of the shape
		int _size; //< number of dimensions
		int *_dimensions; //< number of elements in each dimension, last dimension in bytes
		
		Shape(int size, int *dimensions)
		: _size(size),
		_dimensions(dimensions){
			/* Calculate number of bytes on the shape */
			int n_bytes = _dimensions[_size-1];
			for(int i = size-2; i >= 0; i--){
				n_bytes = dimensions[i] * n_bytes;
			}
			_bytes = n_bytes; 
			_bits = n_bytes * CHAR_BIT; // There may be more than 8 bits in a byte depending on the implementation
		}

		~Shape(){
			delete [] _dimensions;
		}
	} Shape;	


	void *_baseAddress; //< base address of the object [_baseAddress, _endAddress)
	void *_endAddress; //< end address of the object [_baseAddress, _endAddress)
	Shape _shape; //< shape of the object
	RegionSet _homes; //< list of linear regions in each node
	CacheSet _cached; //< list of liner regions replicated in a cache

public:

	MemoryObject(void *baseAddress, int dimensions, int *dimSizes)
	: _baseAddress(baseAddress),
	_shape(dimensions, dimSizes),
	_homes(),
	_cached(){
		_endAddress = static_cast<void*>( static_cast<char*>( _baseAddress) + _shape._bits );
	}

	void *getBaseAddress(){
		return _baseAddress;
	}

	void *getEndAddress(){
		return _endAddress;
	}

	/* Comparison operators for Boost Intrusive AVL Set (OLD) */
	friend bool operator <(const MemoryObject &a, const MemoryObject &b){ return a._baseAddress < b._baseAddress; }
	friend bool operator >(const MemoryObject &a, const MemoryObject &b){ return a._baseAddress > b._baseAddress; }
	friend bool operator ==(const MemoryObject &a, const MemoryObject &b){ return a._baseAddress == b._baseAddress; }
	
	/* Debbuging / Testing */
	friend std::ostream &operator<<(std::ostream &os, const MemoryObject &obj){
		os << "{ MemoryObject: ";
	
		os << "region [" << obj._baseAddress << "," << obj._endAddress << ")";
		
		os << " | ";
		
		os << "shape ";
		for(int i = 0; i < obj._shape._size; i++){
			os << "{" << obj._shape._dimensions[i] << "}";
		}
		os << "-> " << obj._shape._bytes << " bytes";
	
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

#endif //MEMORY_OBJECT_CPP
