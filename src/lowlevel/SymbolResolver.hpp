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
#include <link.h>
#include <string.h>

#include <support/StringLiteral.hpp>

#include <iostream>


class SymbolResolverPrivate {
protected:
	bool _initialized;
	char const *_loaderSharedObjectName;
	char const *_libSharedObjectName;
	
	static SymbolResolverPrivate _singleton;
	
	SymbolResolverPrivate();
	static void initialize();
	
	template <typename RETURN_T, StringLiteral const *NAME, typename... PARAMETERS_T>
	friend class SymbolResolver;
};


template <typename RETURN_T, StringLiteral const *NAME, typename... PARAMETERS_T>
class SymbolResolver {
	
	template <typename LAMBDA_T>
	static int lambdaTrampoline(struct dl_phdr_info *info, size_t size, void *hiddenLambda)
	{
		LAMBDA_T *lambda = (LAMBDA_T *) hiddenLambda;
		return (*lambda)(info, size);
	}
	
	
	static inline void *findNext()
	{
		SymbolResolverPrivate::initialize();
		
		assert(SymbolResolverPrivate::_singleton._loaderSharedObjectName != nullptr);
		assert(SymbolResolverPrivate::_singleton._libSharedObjectName != nullptr);
		
		bool mustSkip = true;
		void *result = nullptr;
		
		auto lambda = [&](struct dl_phdr_info *info, __attribute__((unused)) size_t size) -> int {
			if (result != nullptr) {
				// We already have a result
			} else if (
				(SymbolResolverPrivate::_singleton._loaderSharedObjectName != nullptr)
				&& (strcmp(info->dlpi_name, SymbolResolverPrivate::_singleton._loaderSharedObjectName) == 0)
			) {
				mustSkip = false;
			} else if (
				(SymbolResolverPrivate::_singleton._libSharedObjectName != nullptr)
				&& (strcmp(info->dlpi_name, SymbolResolverPrivate::_singleton._libSharedObjectName) == 0)
			) {
				mustSkip = false;
			} else if (mustSkip) {
				// Skip
			} else {
				void *handle = dlopen(info->dlpi_name, RTLD_LAZY | RTLD_LOCAL);
				if (handle != nullptr) {
					result = dlsym(handle, *NAME);
					dlclose(handle);
				} else {
					std::cerr << "Warning: Could not load " << info->dlpi_name << " to lookup symbol " << *NAME << ": " << dlerror() << std::endl;
				}
			}
			
			return 0;
		};
		
		__attribute__((unused)) int rc = dl_iterate_phdr(lambdaTrampoline<typeof(lambda)>, (void *) &lambda);
		
		return result;
	}
	
	
public:
	typedef RETURN_T (*function_t)(PARAMETERS_T...);
	
	static RETURN_T call(PARAMETERS_T... parameters)
	{
		static function_t symbol = (function_t) dlsym(RTLD_NEXT, *NAME);
		assert(symbol != nullptr);
		
		return (*symbol)(parameters...);
	}
	
	static function_t resolveNext()
	{
		static function_t symbol = (function_t) findNext();
		return symbol;
	}
	
	static RETURN_T callNext(PARAMETERS_T... parameters)
	{
		function_t symbol = resolveNext();
		assert(symbol != nullptr);
		
		return (*symbol)(parameters...);
	}
	
	static function_t getFunction()
	{
		static function_t symbol = (function_t) dlsym(RTLD_NEXT, *NAME);
		assert(symbol != nullptr);
		
		return symbol;
	}
	
	static void *getSymbol()
	{
		static void *symbol = dlsym(RTLD_NEXT, *NAME);
		assert(symbol != nullptr);
		
		return symbol;
	}
	
	
	static RETURN_T globalScopeCall(PARAMETERS_T... parameters)
	{
		static function_t symbol = (function_t) dlsym(RTLD_DEFAULT, *NAME);
		assert(symbol != nullptr);
		
		return (*symbol)(parameters...);
	}
	
	static function_t getGlobalScopeFunction()
	{
		static function_t symbol = (function_t) dlsym(RTLD_DEFAULT, *NAME);
		assert(symbol != nullptr);
		
		return symbol;
	}
	
	static void *getGlobalScopeSymbol()
	{
		static void *symbol = dlsym(RTLD_DEFAULT, *NAME);
		assert(symbol != nullptr);
		
		return symbol;
	}
	
};


#endif // SYMBOL_RESOLVER_HPP
