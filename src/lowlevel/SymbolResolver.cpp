/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif


#include <cassert>
#include <dlfcn.h>
#include <link.h>


#include "SymbolResolver.hpp"


static void dummyFunction()
{
}

SymbolResolverPrivate SymbolResolverPrivate::_singleton;

SymbolResolverPrivate::SymbolResolverPrivate()
{
	_initialized = false;
}


void SymbolResolverPrivate::initialize()
{
	if (_singleton._initialized) {
		return;
	}
	
	Dl_info info;
	
	void *nanos6_loader_symbol = dlsym(RTLD_DEFAULT, "nanos6_start_function_interception");
	assert(nanos6_loader_symbol != nullptr);
	
	int rc = dladdr(nanos6_loader_symbol, &info);
	if (rc != 0) {
		_singleton._loaderSharedObjectName = info.dli_fname;
	}
	
	rc = dladdr((void *) dummyFunction, &info);
	if (rc != 0) {
		_singleton._libSharedObjectName = info.dli_fname;
	}
	
	_singleton._initialized = true;
}

