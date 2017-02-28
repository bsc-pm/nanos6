#ifndef CACHE_TRACKING_OBJECT_HPP
#define CACHE_TRACKING_OBJECT_HPP

#include <utility>
#include <boost/intrusive/treap_set.hpp> 
#include <boost/intrusive/options.hpp>
#include <boost/intrusive/priority_compare.hpp>
#include <DataAccessRange.hpp>

#include "memory/Globals.hpp"
#include "hardware/HardwareInfo.hpp"
#include "CacheTrackingObjectLinkingArtifacts.hpp"
//#include "CacheTrackingSet.hpp"

class CacheTrackingObject;

struct CacheTrackingObjectKey {
    DataAccessRange _range;
    //! Last use to determine evictions
    long unsigned int _lastUse;

    CacheTrackingObjectKey() 
    {}
    CacheTrackingObjectKey(DataAccessRange range, long unsigned int lastUse)
        : _range(range), _lastUse(lastUse)
    {}
    CacheTrackingObjectKey(const CacheTrackingObjectKey &other) 
        : _range(other._range), _lastUse(other._lastUse)
    {}
    DataAccessRange &getAccessRange() {
        return _range;
    }
    DataAccessRange const &getAccessRange() const{
        return _range;
    }
    size_t getSize(){
        return _range.getSize();
    }
    void * getStartAddress() {
        return _range.getStartAddress();
    }
    void * getEndAddress() {
        return _range.getEndAddress();
    }
    void setStartAddress(void * address) {
        _range = DataAccessRange(address, _range.getEndAddress());
    }
    void setEndAddress(void * address) {
        _range = DataAccessRange(_range.getStartAddress(), address);
    }
    long unsigned int getLastUse() {
        return _lastUse;
    }
};


bool operator<(const CacheTrackingObjectKey &a, const CacheTrackingObjectKey &b);
bool operator>(const CacheTrackingObjectKey &a, const CacheTrackingObjectKey &b);
bool priority_order(const CacheTrackingObjectKey &a, const CacheTrackingObjectKey &b);
bool priority_inverse_order(const CacheTrackingObjectKey &a, const CacheTrackingObjectKey &b);

typedef typename CacheTrackingObjectLinkingArtifacts::hook_type CacheTrackingObjectsHook;

struct CacheTrackingObjectHooks {
	CacheTrackingObjectsHook _objectsHook;
};
class CacheTrackingObject: public boost::intrusive::bs_set_base_hook<> { //This is a base hook
private: 
	#if NDEBUG
		typedef boost::intrusive::link_mode<boost::intrusive::normal_link> link_mode_t;
	#else
		typedef boost::intrusive::link_mode<boost::intrusive::safe_link> link_mode_t;
	#endif

public:
    CacheTrackingObjectKey _key;
    boost::intrusive::bs_set_member_hook<link_mode_t> _member_hook;

	//! Links used by the list of objects 
    CacheTrackingObjectHooks _CacheTrackingObjectLinks;

    unsigned int get_priority() const
    {  
        return this->_key._lastUse;
    }

    friend bool operator<(const CacheTrackingObjectKey &a, const CacheTrackingObjectKey &b);
    friend bool operator>(const CacheTrackingObjectKey &a, const CacheTrackingObjectKey &b);
    friend bool priority_order(const CacheTrackingObjectKey &a, const CacheTrackingObjectKey &b);
    friend bool priority_inverse_order(const CacheTrackingObjectKey &a, const CacheTrackingObjectKey &b);

	CacheTrackingObject(DataAccessRange range);
    CacheTrackingObject(CacheTrackingObjectKey key);
	CacheTrackingObject(const CacheTrackingObject &ob);

	DataAccessRange &getAccessRange();
	DataAccessRange const &getAccessRange() const;

	void *getStartAddress();
	void setStartAddress(void * startAddress);

	void *getEndAddress();
    void setEndAddress(void *endAddress);

	size_t getSize();
    
    long unsigned int getLastUse();
    void updateLastUse(long unsigned int value);
};


inline bool operator<(const CacheTrackingObjectKey &a, const CacheTrackingObjectKey &b)
{
    return a._range.getStartAddress() < b._range.getStartAddress();

}
inline bool operator>(const CacheTrackingObjectKey &a, const CacheTrackingObjectKey &b)
{
    return a._range.getStartAddress() > b._range.getStartAddress();

}
inline bool priority_order(const CacheTrackingObjectKey &a, const CacheTrackingObjectKey &b)
{
    return a._lastUse < b._lastUse;
}  //Lower value means higher priority

inline bool priority_inverse_order(const CacheTrackingObjectKey &a, const CacheTrackingObjectKey &b)
{
    return a._lastUse > b._lastUse;
}  //Higher value means higher priority

#endif //CACHE_TRACKING_OBJECT_HPP
