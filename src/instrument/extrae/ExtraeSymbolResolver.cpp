/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2017-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cstdlib>
#include <dlfcn.h>
#include <link.h>

#include "ExtraeSymbolResolver.hpp"
#include "lowlevel/FatalErrorHandler.hpp"


void *ExtraeSymbolResolverBase::loadHandle()
{
	void *handle = nullptr;

	void *f = dlsym(RTLD_DEFAULT, "Extrae_init");
	if (f != nullptr) {
		Dl_info dlInfo = {nullptr, nullptr, nullptr, nullptr};
		int rc = dladdr(f, &dlInfo);
		if (rc == 0)
			FatalErrorHandler::fail("Failed to dladdr extrae symbol");

		handle = dlopen(dlInfo.dli_fname, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
		if (handle == nullptr)
			FatalErrorHandler::fail("Failed to dlopen ", dlInfo.dli_fname, ": ", dlerror());
	} else {
		const char *libname = "libnanostrace.so";
		handle = dlopen(libname, RTLD_LAZY | RTLD_LOCAL);
		if (handle == nullptr)
			FatalErrorHandler::fail("Failed to dlopen ", libname, ": ", dlerror(),
				"\nPlease ensure extrae is in the library search path or preloaded");
	}
	return handle;
}

std::string ExtraeSymbolResolverBase::getSharedObjectPath()
{
	struct link_map *lm = nullptr;

	int rc = dlinfo(getHandle(), RTLD_DI_LINKMAP, &lm);
	if ((rc == 0) && (lm != nullptr)) {
		return lm->l_name;
	} else {
		return std::string();
	}
}

