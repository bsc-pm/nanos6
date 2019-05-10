/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#if HAVE_CONFIG_H
#include <config.h>
#endif

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if HAVE_DLINFO
#include <link.h>
#endif

#include "api/nanos6/debug.h"

#include "main-wrapper.h"
#include "loader.h"


#define MAX_LIB_PATH 8192


__attribute__ ((visibility ("hidden"))) void *_nanos6_lib_handle = NULL;

__attribute__ ((visibility ("hidden"))) int _nanos6_has_started = 0;
int _nanos6_exit_with_error = 0;
char _nanos6_error_text[ERROR_TEXT_SIZE];



static char lib_name[MAX_LIB_PATH+1];


static void _nanos6_loader_set_up_lib_name(char const *variant, char const *path, char const *suffix)
{
	if (path != NULL) {
		strncpy(lib_name, path, MAX_LIB_PATH);
		strncat(lib_name, "/libnanos6-", MAX_LIB_PATH);
	} else {
		strncpy(lib_name, "libnanos6-", MAX_LIB_PATH);
	}
	
	strncat(lib_name, variant, MAX_LIB_PATH);
	strncat(lib_name, ".so", MAX_LIB_PATH);
	
	if (suffix != NULL) {
		strncat(lib_name, ".", MAX_LIB_PATH);
		strncat(lib_name, suffix, MAX_LIB_PATH);
	}
}


static void _nanos6_loader_try_load(_Bool verbose, char const *variant, char const *path)
{
	// A dummy memory reallocation to force preloaded libraries that intercept the memory allocation
	// functions to get initialized, since calls to dlerror cause a call to realloc. If the first call
	// to realloc triggers a dlopen, that call will overwrite the structures used by dlerror and cause
	// an inconsistency and will eventually lead to a crash
	{
		__attribute__((unused)) void *ptr;
		ptr = malloc(0);
		ptr = realloc(NULL, 0);
		free(NULL);
	}
	
	_nanos6_loader_set_up_lib_name(variant, path, SONAME_SUFFIX);
	if (verbose) {
		fprintf(stderr, "Nanos6 loader trying to load: %s\n", lib_name);
	}
	
	_nanos6_lib_handle = dlopen(lib_name, RTLD_LAZY | RTLD_GLOBAL);
	if (_nanos6_lib_handle != NULL) {
		if (verbose) {
			fprintf(stderr, "Successfully loaded: %s\n", nanos6_get_runtime_path());
		}
		return;
	}
	
	if (verbose) {
		fprintf(stderr, "Failed: %s\n", dlerror());
	}
	
	_nanos6_loader_set_up_lib_name(variant, path, SONAME_MAJOR);
	if (verbose) {
		fprintf(stderr, "Nanos6 loader trying to load: %s\n", lib_name);
	}
	
	_nanos6_lib_handle = dlopen(lib_name, RTLD_LAZY | RTLD_GLOBAL);
	if (_nanos6_lib_handle != NULL) {
		if (verbose) {
			fprintf(stderr, "Successfully loaded: %s\n", nanos6_get_runtime_path());
		}
		return;
	}
	
	if (verbose) {
		fprintf(stderr, "Failed: %s\n", dlerror());
	}
}


static void _nanos6_loader_try_load_without_major(_Bool verbose, char const *variant, char const *path)
{
	_nanos6_loader_set_up_lib_name(variant, path, NULL);
	if (verbose) {
		fprintf(stderr, "Nanos6 loader trying to load: %s\n", lib_name);
	}
	
	_nanos6_lib_handle = dlopen(lib_name, RTLD_LAZY | RTLD_GLOBAL);
	if (_nanos6_lib_handle != NULL) {
		if (verbose) {
			fprintf(stderr, "Successfully loaded: %s\n", nanos6_get_runtime_path());
		}
		return;
	}
}


__attribute__ ((visibility ("hidden"), constructor)) void _nanos6_loader(void)
{
	if (_nanos6_lib_handle != NULL) {
		return;
	}
	
	_Bool verbose = (getenv("NANOS6_LOADER_VERBOSE") != NULL);
	
	// Check the name of the replacement library
	char const *variant = getenv("NANOS6");
	if (variant == NULL) {
		variant = "optimized";
	}
	
	if (verbose) {
		fprintf(stderr, "Nanos6 loader using variant: %s\n", variant);
	}
	
	char *lib_path = getenv("NANOS6_LIBRARY_PATH");
	if (lib_path != NULL) {
		if (verbose) {
			fprintf(stderr, "Nanos6 loader using path from NANOS6_LIBRARY_PATH: %s\n", lib_path);
		}
	}
	
	// Try the global or the NANOS6_LIBRARY_PATH scope
	_nanos6_loader_try_load(verbose, variant, lib_path);
	if (_nanos6_lib_handle != NULL) {
		return;
	}
	
	// Attempt to load it from the same path as this library
	Dl_info di;
	int rc = dladdr((void *)_nanos6_loader, &di);
	assert(rc != 0);
	
	lib_path = strdup(di.dli_fname);
	for (int i = strlen(lib_path); i > 0; i--) {
		if (lib_path[i] == '/') {
			lib_path[i] = 0;
			break;
		}
	}
	_nanos6_loader_try_load(verbose, variant, lib_path);
	
	// Check if this is a disabled variant
	if (_nanos6_lib_handle != NULL) {
		void *disabled_symbol = dlsym(_nanos6_lib_handle, "nanos6_disabled_variant");
		if (disabled_symbol != NULL) {
			snprintf(_nanos6_error_text, ERROR_TEXT_SIZE, "This installation of Nanos6 does not include the %s variant.", variant);
			_nanos6_exit_with_error = 1;
			
			return;
		}
	}
	
	if (_nanos6_lib_handle == NULL) {
		snprintf(_nanos6_error_text, ERROR_TEXT_SIZE, "Nanos6 loader failed to load the runtime library.");
		_nanos6_exit_with_error = 1;
		
		//
		// Diagnose the problem
		//
		
		// Check the variant
		if (verbose) {
			fprintf(stderr, "Checking if the variant was not correct\n");
		}
		
		_nanos6_loader_try_load(verbose, "optimized", getenv("NANOS6_LIBRARY_PATH"));
		if (_nanos6_lib_handle == NULL) {
			_nanos6_loader_try_load(verbose, "optimized", lib_path);
		}
		if (_nanos6_lib_handle != NULL) {
			fprintf(stderr, "Error: the %s variant of the runtime is not available in this installation.\n", variant);
			fprintf(stderr, "\tPlease check that the NANOS6 environment variable is valid.\n");
			
			dlclose(_nanos6_lib_handle);
			_nanos6_lib_handle = NULL;
			
			return;
		}
		
		// Check for version mismatch
		if (verbose) {
			fprintf(stderr, "Checking for a mismatch between the linked version and the installed version\n");
		}
		
		_nanos6_loader_try_load_without_major(verbose, variant, getenv("NANOS6_LIBRARY_PATH"));
		if (_nanos6_lib_handle == NULL) {
			_nanos6_loader_try_load_without_major(verbose, variant, lib_path);
		}
		if (_nanos6_lib_handle != NULL) {
			fprintf(stderr, "Error: there is a mismatch between the installed runtime so version and the linked so version\n");
			fprintf(stderr, "\tExpected so version: %s or at least %s\n", SONAME_SUFFIX, SONAME_MAJOR);
			fprintf(stderr, "\tFound instead this so: %s\n", nanos6_get_runtime_path());
			fprintf(stderr, "\tPlease recompile your application.\n");
			
			dlclose(_nanos6_lib_handle);
			_nanos6_lib_handle = NULL;
			
			return;
		}
		
		if (getenv("NANOS6") != NULL) {
			fprintf(stderr, "Please check that the value of the NANOS6 environment variable is correct and set the NANOS6_LIBRARY_PATH environment variable if the runtime is installed in a different location than the loader.\n");
		} else {
			fprintf(stderr, "Please set or check the NANOS6_LIBRARY_PATH environment variable if the runtime is installed in a different location than the loader.\n");
		}
	}
}


#pragma GCC visibility push(default)

char const *nanos6_get_runtime_path(void)
{
#if HAVE_DLINFO
	if (_nanos6_lib_handle == NULL) {
		_nanos6_loader();
	}
	
	static char const *lib_path = NULL;
	static int initialized = 0;
	
	if (initialized == 0) {
		void *symbol = dlsym(_nanos6_lib_handle, "nanos6_preinit");
		if (symbol == NULL) {
			lib_path = strdup(dlerror());
		} else {
			Dl_info di;
			int rc = dladdr(symbol, &di);
			
			if (rc != 0) {
				lib_path = strdup(di.dli_fname);
			} else {
				lib_path = strdup(dlerror());
			}
		}
		
		initialized = 1;
	}
	
	return lib_path;
#else
	return "not available";
#endif
}

#pragma GCC visibility pop
