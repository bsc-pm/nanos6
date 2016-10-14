#ifndef CACHE_OBJECT_HPP
#define CACHE_OBJECT_HPP

#include <boost/intrusive/avl_set.hpp>
#include "memory/Region.hpp"

class CopyObject: public boost::intrusive::avl_set_base_hook<> {
private: 
	Region _region;

	unsigned int _version;
	// Cache of the latest version	

public:
	CopyObject(void *baseAddress, size_t size)
	: _region(baseAddress, size),
	_version(0){
	
	}

	void *getBaseAddress(){
		return _region._baseAddress;
	}

	void *getEndAddress(){
		return _region._endAddress;
	}

	size_t getSize(){
		return _region._size;
	}

	int getVersion(){
		return _version;
	}

	void setVersion(int version){
		_version = version;
	}

	void incrementVersion(){
		_version++;
	} 
	
	/* Key for Boost Intrusive AVL Set */
    struct key_value
    {
        typedef void *type;

        const type &operator()(const CopyObject &obj){
            return obj._region._baseAddress;
        }
    };

    friend key_value;

};

#endif //CACHE_OBJECT_HPP
