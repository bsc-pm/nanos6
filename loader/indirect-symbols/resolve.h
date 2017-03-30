#ifndef RESOLVE_H
#define RESOLVE_H


#include "loader.h"

#include "common/api/nanos6.h"

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>


static void *_nanos6_resolve_symbol(char const *fname, char const *area, char const *fallback)
{
	if (__builtin_expect(_nanos6_lib_handle == NULL, 0)) {
		_nanos6_loader();
		if (__builtin_expect(_nanos6_lib_handle == NULL, 0)) {
			fprintf(stderr, "Nanos 6 loader error: call to %s before library initialization\n", fname);
			abort();
		}
	}
	
	void *symbol = dlsym(_nanos6_lib_handle, fname);
	if ((symbol == NULL) && (fallback != NULL)) {
		symbol = dlsym(_nanos6_lib_handle, fallback);
		if (symbol != NULL) {
			fprintf(stderr, "Nanos 6 loader warning: %s runtime function %s is undefined in '%s' falling back to function %s instead\n", area, fname, _nanos6_lib_filename, fallback);
		}
	}
	if (symbol == NULL) {
		fprintf(stderr, "Nanos 6 loader error: %s runtime function %s is undefined in '%s'\n", area, fname, _nanos6_lib_filename);
		abort();
	}
	
	return symbol;
}


static void *_nanos6_resolve_symbol_with_local_fallback(char const *fname, char const *area, void *fallback, char const *fallback_name)
{
	if (__builtin_expect(_nanos6_lib_handle == NULL, 0)) {
		_nanos6_loader();
		if (__builtin_expect(_nanos6_lib_handle == NULL, 0)) {
			fprintf(stderr, "Nanos 6 loader error: call to %s before library initialization\n", fname);
			abort();
		}
	}
	
	void *symbol = dlsym(_nanos6_lib_handle, fname);
	if (symbol == NULL) {
		symbol = fallback;
		if (symbol != NULL) {
			fprintf(stderr, "Nanos 6 loader warning: %s runtime function %s is undefined in '%s' falling back to function %s instead\n", area, fname, _nanos6_lib_filename, fallback_name);
		}
	}
	
	return symbol;
}


#endif /* RESOLVE_H */
