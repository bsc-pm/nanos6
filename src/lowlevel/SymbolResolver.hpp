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

#include <support/StringLiteral.hpp>


template <typename RETURN_T, StringLiteral const *NAME, typename... PARAMETERS_T>
class SymbolResolver {
public:
	typedef RETURN_T (*function_t)(PARAMETERS_T...);
	
	static RETURN_T call(PARAMETERS_T... parameters)
	{
		static function_t symbol = (function_t) dlsym(RTLD_NEXT, *NAME);
		assert(symbol != nullptr);
		
		return (*symbol)(parameters...);
	}
	
	static RETURN_T globalScopeCall(PARAMETERS_T... parameters)
	{
		static function_t symbol = (function_t) dlsym(RTLD_DEFAULT, *NAME);
		assert(symbol != nullptr);
		
		return (*symbol)(parameters...);
	}
};


#endif // SYMBOL_RESOLVER_HPP
