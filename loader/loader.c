/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2021 Barcelona Supercomputing Center (BSC)
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

#include "config-parser.h"
#include "error.h"
#include "main-wrapper.h"
#include "loader.h"


#define MAX_LIB_PATH 8192


__attribute__ ((visibility ("hidden"))) void *_nanos6_lib_handle = NULL;
int _nanos6_exit_with_error = 0;
char _nanos6_error_text[ERROR_TEXT_SIZE];

static char _lib_name[MAX_LIB_PATH+1];
static char _lib_name_aux[MAX_LIB_PATH+1];

static void _nanos6_loader_set_up_lib_name(char const *optimization, char const *dependencies, char const *instrument, char const *path, char const *suffix)
{
	if (strcmp(instrument, "none")) {
		snprintf(_lib_name_aux, MAX_LIB_PATH, "libnanos6-%s-%s-%s.so", optimization, dependencies, instrument);
	} else {
		snprintf(_lib_name_aux, MAX_LIB_PATH, "libnanos6-%s-%s.so", optimization, dependencies);
	}

	if (path != NULL) {
		if (suffix == NULL) {
			snprintf(_lib_name, MAX_LIB_PATH, "%s/%s", path, _lib_name_aux);
		} else {
			snprintf(_lib_name, MAX_LIB_PATH, "%s/%s.%s", path, _lib_name_aux, suffix);
		}
	} else {
		if (suffix == NULL) {
			snprintf(_lib_name, MAX_LIB_PATH, "%s", _lib_name_aux);
		} else {
			snprintf(_lib_name, MAX_LIB_PATH, "%s.%s", _lib_name_aux, suffix);
		}
	}
}

static void _nanos6_loader_try_load(_Bool verbose, char const *optimization, char const *dependencies, char const *instrument, char const *path)
{
	_nanos6_loader_set_up_lib_name(optimization, dependencies, instrument, path, SONAME_SUFFIX);
	if (verbose) {
		fprintf(stderr, "Nanos6 loader trying to load: %s\n", _lib_name);
	}

	_nanos6_lib_handle = dlopen(_lib_name, RTLD_LAZY | RTLD_GLOBAL);
	if (_nanos6_lib_handle != NULL) {
		if (verbose) {
			fprintf(stderr, "Successfully loaded: %s\n", nanos6_get_runtime_path());
		}
		return;
	}

	if (verbose) {
		fprintf(stderr, "Failed: %s\n", dlerror());
	}

	_nanos6_loader_set_up_lib_name(optimization, dependencies, instrument, path, SONAME_MAJOR);
	if (verbose) {
		fprintf(stderr, "Nanos6 loader trying to load: %s\n", _lib_name);
	}

	_nanos6_lib_handle = dlopen(_lib_name, RTLD_LAZY | RTLD_GLOBAL);
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

static void _nanos6_loader_try_load_without_major(_Bool verbose, char const *optimization, char const *dependencies, char const *instrument, char const *path)
{
	_nanos6_loader_set_up_lib_name(optimization, dependencies, instrument, path, NULL);
	if (verbose) {
		fprintf(stderr, "Nanos6 loader trying to load: %s\n", _lib_name);
	}

	_nanos6_lib_handle = dlopen(_lib_name, RTLD_LAZY | RTLD_GLOBAL);
	if (_nanos6_lib_handle != NULL) {
		if (verbose) {
			fprintf(stderr, "Successfully loaded: %s\n", nanos6_get_runtime_path());
		}
		return;
	}
}

static int _nanos6_check_disabled_variant(char const *optimization, char const *dependencies, char const *instrument, char const *common_error)
{
	assert(_nanos6_lib_handle != NULL);
	assert(optimization != NULL);
	assert(dependencies != NULL);
	assert(instrument != NULL);

	void *disabled_symbol = dlsym(_nanos6_lib_handle, "nanos6_disabled_variant");
	if (disabled_symbol != NULL) {
		fprintf(stderr, "Error: %s\n", common_error);
		fprintf(stderr, "This installation has disabled the '%s' variant with '%s' dependencies and '%s' instrumentation.\n", optimization, dependencies, instrument);
		return -1;
	}
	return 0;
}

static int _nanos6_loader_impl(void)
{
	if (_nanos6_lib_handle != NULL)
		return 0;

	if (_nanos6_loader_parse_config())
		return -1;

	// Enable discrete dependencies by default
	char const *default_dependencies = "discrete";

	_Bool debug = _config.debug;
	_Bool verbose = _config.verbose;

	// Check the optimization variant to load
	char const *optimization = (debug) ? "debug" : "optimized";

	// Check the instrumentation variant to load
	char const *instrument = _config.instrument;
	if (instrument == NULL) {
		// No instrumentation by default
		instrument = "none";
	}

	char const *dependencies = _config.dependencies;
	if (dependencies == NULL) {
		dependencies = default_dependencies;
	}

	if (verbose) {
		fprintf(stderr, "Nanos6 loader using '%s' variant with '%s' dependencies and '%s' instrumentation.\n", optimization, dependencies, instrument);
	}

	char *lib_path = _config.library_path;
	if (lib_path != NULL) {
		if (verbose) {
			fprintf(stderr, "Nanos6 loader using library path from loader.library_path: %s\n", lib_path);
		}
	}

	char const *common_error = "Nanos6 loader failed to load the runtime library.";

	// Try the global or the loader.library_path scope
	_nanos6_loader_try_load(verbose, optimization, dependencies, instrument, lib_path);
	if (_nanos6_lib_handle != NULL) {
		// Check if this is a disabled variant
		return _nanos6_check_disabled_variant(optimization, dependencies, instrument, common_error);
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
	_nanos6_loader_try_load(verbose, optimization, dependencies, instrument, lib_path);
	if (_nanos6_lib_handle != NULL) {
		free(lib_path);
		// Check if this is a disabled variant
		return _nanos6_check_disabled_variant(optimization, dependencies, instrument, common_error);
	}

	fprintf(stderr, "Error: %s\n", common_error);

	//
	// Diagnose the problem
	//

	// Check the variant
	if (verbose) {
		fprintf(stderr, "Diagnosing the loader issue...\n");
		fprintf(stderr, "Checking if the variant was not correct\n");
	}

	_nanos6_loader_try_load(verbose, "optimized", default_dependencies, "none", _config.library_path);
	if (_nanos6_lib_handle == NULL) {
		_nanos6_loader_try_load(verbose, "optimized", default_dependencies, "none", lib_path);
	}
	if (_nanos6_lib_handle != NULL) {
		fprintf(stderr, "The '%s' runtime variant with '%s' dependencies and '%s' instrumentation does not exist.\n", optimization, dependencies, instrument);
		fprintf(stderr, "Please check that the version.dependencies and version.instrument configuration options are valid.\n");

		dlclose(_nanos6_lib_handle);
		_nanos6_lib_handle = NULL;
		free(lib_path);

		return -1;
	}

	// Check for version mismatch
	if (verbose) {
		fprintf(stderr, "Checking for a mismatch between the linked version and the installed version\n");
	}

	_nanos6_loader_try_load_without_major(verbose, optimization, dependencies, instrument, _config.library_path);
	if (_nanos6_lib_handle == NULL) {
		_nanos6_loader_try_load_without_major(verbose, optimization, dependencies, instrument, lib_path);
	}
	if (_nanos6_lib_handle != NULL) {
		fprintf(stderr, "There is a mismatch between the installed runtime so version and the linked so version\n");
		fprintf(stderr, "\tExpected so version: %s or at least %s\n", SONAME_SUFFIX, SONAME_MAJOR);
		fprintf(stderr, "\tFound instead this so: %s\n", nanos6_get_runtime_path());
		fprintf(stderr, "\tPlease recompile your application.\n");

		dlclose(_nanos6_lib_handle);
		_nanos6_lib_handle = NULL;
		free(lib_path);

		return -1;
	}

	if (_config.dependencies != NULL) {
		fprintf(stderr, "Please check that the value of the version.dependencies configuration option is correct and set the loader.library_path option if the runtime is installed in a different location than the loader.\n");
	} else if (_config.instrument != NULL) {
		fprintf(stderr, "Please check that the value of the version.instrument configuration option is correct and set the loader.library_path option if the runtime is installed in a different location than the loader.\n");
	} else {
		fprintf(stderr, "Please set or check the loader.library_path if the runtime is installed in a different location than the loader.\n");
	}

	free(lib_path);
	return -1;
}

__attribute__ ((visibility ("hidden"), constructor)) void _nanos6_loader(void)
{
	if (_nanos6_loader_impl()) {
		handle_error();
	}
}

__attribute__ ((visibility ("hidden"), destructor)) void _nanos6_loader_destructor(void)
{
	_nanos6_loader_free_config();
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
