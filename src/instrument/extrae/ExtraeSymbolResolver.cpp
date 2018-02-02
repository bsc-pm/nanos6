/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <link.h>

#include "ExtraeSymbolResolver.hpp"

#include <cstdlib>
#include <iostream>


ExtraeSymbolResolverBase ExtraeSymbolResolverBase::_singleton;


ExtraeSymbolResolverBase::ExtraeSymbolResolverBase()
{
	void *f = dlsym(RTLD_DEFAULT, "Extrae_init");
	if (f != nullptr) {
		Dl_info dlInfo = {nullptr, nullptr, nullptr, nullptr};
		__attribute__((unused)) int rc = dladdr(f, &dlInfo);
		
		_handle = dlopen(dlInfo.dli_fname, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
		if (_handle == nullptr) {
			std::cerr << "Error: failed to get a handle to \"" << dlInfo.dli_fname << "\": " << dlerror() << std::endl;
			abort();
		}
	} else {
		_handle = dlopen("libnanostrace.so", RTLD_LAZY | RTLD_LOCAL);
		if (_handle == nullptr) {
			std::cerr << "Error: failed to load " << dlerror() << std::endl;
			std::cerr << "\tPlease make sure that extrae is in the library search path or preloaded." << std::endl;
			abort();
		}
	}
}


std::string ExtraeSymbolResolverBase::getSharedObjectPath()
{
	struct link_map *lm = nullptr;
	
	int rc = dlinfo(_singleton._handle, RTLD_DI_LINKMAP, &lm);
	if ((rc == 0) && (lm != nullptr)) {
		return lm->l_name;
	} else {
		return std::string();
	}
}

