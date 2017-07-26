/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef CONST_PROPAGATOR_HPP
#define CONST_PROPAGATOR_HPP


#include <type_traits>


namespace const_propagator_internals {
	template<typename T, bool MAKE_CONST = false>
	struct ConstPropagatorHelper {
		typedef T type;
	};
	
	template<typename T>
	struct ConstPropagatorHelper<T, true> {
		typedef typename std::add_const<T>::type type;
	};
}


template<typename SOURCE_T, typename TARGET_T>
struct ConstPropagator {
	typedef typename const_propagator_internals::ConstPropagatorHelper<
		TARGET_T,
		std::is_const<SOURCE_T>::value
	>::type type;
};


#endif // CONST_PROPAGATOR_HPP
