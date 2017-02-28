#ifndef CACHE_TRACKING_OBJECT_HPP
#define CACHE_TRACKING_OBJECT_HPP

#include <utility>
#include <boost/intrusive/treap_set.hpp> 
#include <boost/intrusive/options.hpp>
#include <DataAccessRange.hpp>

#include "memory/Globals.hpp"
#include "hardware/HardwareInfo.hpp"

class CacheTrackingObject;

bool operator<(const CacheTrackingObject &a, const CacheTrackingObject &b);
bool operator>(const CacheTrackingObject &a, const CacheTrackingObject &b);
bool priority_order(const CacheTrackingObject &a, const CacheTrackingObject &b);
bool priority_inverse_order(const CacheTrackingObject &a, const CacheTrackingObject &b);

class CacheTrackingObject: public boost::intrusive::bs_set_base_hook<> { //This is a base hook
private: 
    //! Last use to determine evictions
    long unsigned int _lastUse;

	#if NDEBUG
		typedef boost::intrusive::link_mode<boost::intrusive::normal_link> link_mode_t;
	#else
		typedef boost::intrusive::link_mode<boost::intrusive::safe_link> link_mode_t;
	#endif

public:
	DataAccessRange _range;

    boost::intrusive::bs_set_member_hook<> _member_hook;

    unsigned int get_priority() const
    {  
        return this->_lastUse;
    }

    friend bool operator<(const CacheTrackingObject &a, const CacheTrackingObject &b);
    friend bool operator>(const CacheTrackingObject &a, const CacheTrackingObject &b);
    friend bool priority_order(const CacheTrackingObject &a, const CacheTrackingObject &b);
    friend bool priority_inverse_order(const CacheTrackingObject &a, const CacheTrackingObject &b);
    //struct default_priority
    //{
    //    bool operator()(const CacheTrackingObject &a, const CacheTrackingObject &b) const
    //    { return priority_order(a, b); }
    //};

//struct inverse_priority
//{
//   bool operator()(const CacheTrackingTreapNode &a, const CacheTrackingTreapNode &b) const
//   {  return priority_inverse_order(a, b); }
//};

	CacheTrackingObject(DataAccessRange range);
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


bool operator<(const CacheTrackingObject &a, const CacheTrackingObject &b)
{
    return a._range.getStartAddress() < b._range.getStartAddress();

}
bool operator>(const CacheTrackingObject &a, const CacheTrackingObject &b)
{
    return a._range.getStartAddress() > b._range.getStartAddress();

}
bool priority_order(const CacheTrackingObject &a, const CacheTrackingObject &b)
{
    return a._lastUse < b._lastUse;
}  //Lower value means higher priority

bool priority_inverse_order(const CacheTrackingObject &a, const CacheTrackingObject &b)
{
    return a._lastUse > b._lastUse;
}  //Higher value means higher priority

#endif //CACHE_TRACKING_OBJECT_HPP
