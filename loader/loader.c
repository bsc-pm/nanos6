/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "main-wrapper.h"
#include "loader.h"


__attribute__ ((visibility ("hidden"))) void *_nanos6_lib_handle = NULL;
__attribute__ ((visibility ("hidden"))) char const *_nanos6_lib_filename = NULL;


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
	size_t variant_length = strlen(variant);
	
	if (verbose) {
		fprintf(stderr, "Nanos6 loader using variant: %s\n", variant);
	}
	
	char *lib_path = getenv("NANOS6_LIBRARY_PATH");
	size_t path_length = 0;
	if (lib_path != NULL) {
		path_length = strlen(lib_path) + 1;
		if (verbose) {
			fprintf(stderr, "Nanos6 loader using path: %s\n", lib_path);
		}
	}
	
	size_t fixed_length = strlen("libnanos6-.so");
	
	// Load the library in the global scope
	{
		size_t lib_name_length = path_length + fixed_length + variant_length;
		char lib_name[lib_name_length+1];
		
		if (lib_path == NULL) {
			snprintf(lib_name, lib_name_length+1, "libnanos6-%s.so", variant);
		} else {
			snprintf(lib_name, lib_name_length+1, "%s/libnanos6-%s.so", lib_path, variant);
		}
		if (verbose) {
			fprintf(stderr, "Nanos6 loader loading: %s\n", lib_name);
		}
		_nanos6_lib_handle = dlopen(lib_name, RTLD_LAZY | RTLD_GLOBAL);
		if (verbose && (_nanos6_lib_handle == NULL)) {
			fprintf(stderr, "Nanos6 loader failed to load %s\n", dlerror());
		}
		
		if (_nanos6_lib_handle != NULL) {
			_nanos6_lib_filename = strdup(lib_name);
		}
	}
	
	// Attempt to load it from the same path as this library
	if (_nanos6_lib_handle == NULL) {
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
		path_length = strlen(lib_path) + 1;
		
		size_t lib_name_length = path_length + fixed_length + variant_length;
		char lib_name[lib_name_length+1];
		
		if (lib_path == NULL) {
			snprintf(lib_name, lib_name_length+1, "libnanos6-%s.so", variant);
		} else {
			snprintf(lib_name, lib_name_length+1, "%s/libnanos6-%s.so", lib_path, variant);
		}
		free(lib_path);
		
		if (verbose) {
			fprintf(stderr, "Nanos6 loader loading: %s\n", lib_name);
		}
		_nanos6_lib_handle = dlopen(lib_name, RTLD_LAZY | RTLD_GLOBAL);
		if (verbose && (_nanos6_lib_handle == NULL)) {
			fprintf(stderr, "Nanos6 loader failed to load %s\n", dlerror());
		}
		
		if (_nanos6_lib_handle != NULL) {
			_nanos6_lib_filename = strdup(lib_name);
		}
	}
	
	if (_nanos6_lib_handle == NULL) {
		fprintf(stderr, "Nanos6 loader failed to load the runtime library. ");
		if (getenv("NANOS6") != NULL) {
			fprintf(stderr, "Please check that the value of the NANOS6 environment variable is correct and set the NANOS6_LIBRARY_PATH environment variable if the runtime is installed in a different location than the loader.\n");
		} else {
			fprintf(stderr, "Please set or check the NANOS6_LIBRARY_PATH environment variable if the runtime is installed in a different location than the loader.\n");
		}
		abort();
	}
	assert(_nanos6_lib_handle != NULL);
}


