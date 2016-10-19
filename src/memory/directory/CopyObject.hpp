#ifndef CACHE_OBJECT_HPP
#define CACHE_OBJECT_HPP

#include <boost/intrusive/avl_set.hpp>
#include "memory/Region.hpp"
#include "memory/cache/GenericCache.hpp"

class CopyObject: public boost::intrusive::avl_set_base_hook<> {
private: 
	Region _region;

	unsigned int _version;
	std::set<GenericCache *> _caches;

public:
	CopyObject(void *baseAddress, size_t size)
	: _region(baseAddress, size),
	_caches(),
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

	void addCache(GenericCache *cache){
		_caches.add(cache);
	}

	void removeCache(GenericCache *cache){
		_caches.erase(_caches.find(cache));
	}	
 
	bool isInCache(GenericCache *cache){
		return _caches.count(cache) != 0;	
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
