/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef SYMBOL_RESOLVER_HPP
#define SYMBOL_RESOLVER_HPP


#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif


#include <cassert>
#include <dlfcn.h>


template <typename RETURN_T, typename... PARAMETERS_T>
class SymbolResolver {
public:
	typedef RETURN_T (*function_t)(PARAMETERS_T...);
	
	static RETURN_T call(char const *name, PARAMETERS_T... parameters)
	{
		static function_t symbol = (function_t) dlsym(RTLD_NEXT, name);
		assert(symbol != nullptr);
		
		return (*symbol)(parameters...);
	}
};


#endif // SYMBOL_RESOLVER_HPP
