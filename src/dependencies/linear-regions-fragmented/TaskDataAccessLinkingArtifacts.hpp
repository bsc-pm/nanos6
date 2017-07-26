/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TASK_DATA_ACCESS_LINKING_ARTIFACTS_HPP
#define TASK_DATA_ACCESS_LINKING_ARTIFACTS_HPP


#include <boost/intrusive/avl_set_hook.hpp>


struct DataAccess;


struct TaskDataAccessLinkingArtifacts {
	#if NDEBUG
		typedef boost::intrusive::link_mode<boost::intrusive::normal_link> link_mode_t;
	#else
		typedef boost::intrusive::link_mode<boost::intrusive::safe_link> link_mode_t;
	#endif
	
	typedef boost::intrusive::avl_set_member_hook<link_mode_t> hook_type;
	typedef hook_type* hook_ptr;
	typedef const hook_type* const_hook_ptr;
	typedef DataAccess value_type;
	typedef value_type* pointer;
	typedef const value_type* const_pointer;
	
	static inline constexpr hook_ptr to_hook_ptr (value_type &value);
	static inline constexpr const_hook_ptr to_hook_ptr(const value_type &value);
	static inline pointer to_value_ptr(hook_ptr n);
	static inline const_pointer to_value_ptr(const_hook_ptr n);
};


#endif // TASK_DATA_ACCESS_LINKING_ARTIFACTS_HPP
