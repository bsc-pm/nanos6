#ifndef CACHE_OBJECT_HPP
#define CACHE_OBJECT_HPP

#include <boost/intrusive/avl_set.hpp>

class CacheObject: public boost::intrusive::avl_set_base_hook<> {
private: 
	void *_baseAddress;
	int _size;	

	int _version;
	// Cache of the latest version	

public:
	CacheObject(void *baseAddress, int size, int version)
	: _baseAddress( baseAddress ),
	_size( size ), 
	_version( version ){
		
	}

	/* Comparison operators for Boost Intrusive AVL Set (OLD) */
    friend bool operator <(const CacheObject &a, const CacheObject &b) { return a._baseAddress < b._baseAddress; }
    friend bool operator >(const CacheObject &a, const CacheObject &b) { return a._baseAddress > b._baseAddress; }
    friend bool operator ==(const CacheObject &a, const CacheObject &b) { return a._baseAddress == b._baseAddress; }

	/* Debugging / Testing */
    friend std::ostream& operator<<(std::ostream &os, const CacheObject &obj){
        void *end = static_cast<char*>(obj._baseAddress) + obj._size;
		os << "{ CacheObject: ";
		
		os << "region [" << obj._baseAddress << "-" << end <<")";
		
		os <<" | ";

		os << "version: " << obj._version;

		os << " }";
        return os;
    }

	/* Key for Boost Intrusive AVL Set */
    struct key_value
    {
        typedef void *type;

        const type &operator()(const CacheObject &obj){
            return obj._baseAddress;
        }
    };

    friend key_value;

};

#endif //CACHE_OBJECT_HPP
