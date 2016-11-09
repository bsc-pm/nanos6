#ifndef CACHE_OBJECT_HPP
#define CACHE_OBJECT_HPP

#include <DataAccessRange.hpp>
#include <boost/intrusive/avl_set_hook.hpp>
#include <boost/intrusive/parent_from_member.hpp>

#include "memory/Globals.hpp"

class CopyObject {
private: 
	DataAccessRange _range;
	unsigned int _version;
	cache_mask _caches;

	#if NDEBUG
		typedef boost::intrusive::link_mode<boost::intrusive::normal_link> link_mode_t;
	#else
		typedef boost::intrusive::link_mode<boost::intrusive::safe_link> link_mode_t;
	#endif

public:
	

	typedef boost::intrusive::avl_set_member_hook<link_mode_t> hook_type;
	
	hook_type _hook;

	CopyObject(void *startAddress, void *endAddress);
	CopyObject(void *startAddress, size_t size);

	DataAccessRange &getAccessRange();
	DataAccessRange const &getAccessRange() const;

	void *getStartAddress();
	void setStartAddress(void * startAddress);

	void *getEndAddress();
    void setEndAddress(void *endAddress);

	size_t getSize();

	int getVersion();
	void setVersion(int version);
	void incrementVersion();
	
	void addCache(int id);
	void removeCache(int id);
	bool testCache(int id);
	bool anyCache();
	int countCaches();
};

struct CopyObjectLinkingArtifacts {

	
	typedef CopyObject::hook_type hook_type;
	typedef hook_type* hook_ptr;
	typedef const hook_type* const_hook_ptr;
	typedef CopyObject value_type;
	typedef value_type* pointer;
	typedef const value_type* const_pointer;
	
	static inline constexpr hook_ptr to_hook_ptr (value_type &value){
		return &value._hook;
	}

	static inline constexpr const_hook_ptr to_hook_ptr(const value_type &value){
		return &value._hook;
	}
	
	static inline pointer to_value_ptr(hook_ptr n){
		return (pointer)
			boost::intrusive::get_parent_from_member<CopyObject>(
				n,
				&CopyObject::_hook
			);
	}

	static inline const_pointer to_value_ptr(const_hook_ptr n)
	{
		return (const_pointer)
			boost::intrusive::get_parent_from_member<CopyObject>(
				n,
				&CopyObject::_hook
			);
	}

};


#endif //CACHE_OBJECT_HPP
