#ifndef REGION_HPP
#define REGION_HPP

#include <DataAccessRange.hpp>
#include <boost/intrusive/avl_set_hook.hpp>
#include <boost/intrusive/parent_from_member.hpp>

class MemoryPageObject{
	
private:
    int _location; //< memory node where the page resides, -1 means memory node where the task is going to be executed.


	#if NDEBUG
		typedef boost::intrusive::link_mode<boost::intrusive::normal_link> link_mode_t;
	#else
		typedef boost::intrusive::link_mode<boost::intrusive::safe_link> link_mode_t;
	#endif

public:
	DataAccessRange _range;

	typedef boost::intrusive::avl_set_member_hook<link_mode_t> hook_type;
	
	hook_type _hook;

	MemoryPageObject( void *baseAddress, size_t size, int location );
	
	DataAccessRange &getAccessRange();
	DataAccessRange const &getAccessRange() const;

	void *getStartAddress();
	void setStartAddress(void *address);
	void *getEndAddress();
	void setEndAddress(void *address);
	size_t getSize();
	int getLocation();

};

struct MemoryPageObjectLinkingArtifacts {

	
	typedef MemoryPageObject::hook_type hook_type;
	typedef hook_type* hook_ptr;
	typedef const hook_type* const_hook_ptr;
	typedef MemoryPageObject value_type;
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
			boost::intrusive::get_parent_from_member<MemoryPageObject>(
				n,
				&MemoryPageObject::_hook
			);
	}

	static inline const_pointer to_value_ptr(const_hook_ptr n)
	{
		return (const_pointer)
			boost::intrusive::get_parent_from_member<MemoryPageObject>(
				n,
				&MemoryPageObject::_hook
			);
	}

};



#endif //REGION_HPP
