#ifndef RESOLVE_H
#define RESOLVE_H


#include "loader.h"

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>


#define RESOLVE_API_FUNCTION(fname, area, fallback) \
void (*_##fname##_resolver(void)) (void) \
{ \
	if (__builtin_expect(_nanos6_lib_handle == NULL, 0)) { \
		fprintf(stderr, "Nanos 6 loader error: call to " #fname " before library initialization\n"); \
		abort(); \
	} \
	\
	void *symbol = (void (*)(void)) dlsym(_nanos6_lib_handle, #fname); \
	if ((symbol == NULL) && (#fallback != NULL)) { \
		symbol = (void (*)(void)) dlsym(_nanos6_lib_handle, #fallback); \
		if (symbol != NULL) { \
			fprintf(stderr, "Nanos 6 loader warning: " #area " runtime function " #fname " is undefined in '%s' falling back to function " #fallback " instead\n", _nanos6_lib_filename); \
		} \
	} \
	if (symbol == NULL) { \
		fprintf(stderr, "Nanos 6 loader error: " #area " runtime function " #fname " is undefined in '%s'\n", _nanos6_lib_filename); \
		abort(); \
	} \
	\
	return (void (*)(void)) symbol; \
} \
\
void *fname() __attribute__ (( ifunc("_" #fname "_resolver") ))


#define RESOLVE_API_FUNCTION_WITH_LOCAL_FALLBACK(fname, area, fallback) \
void (*_##fname##_resolver(void)) (void) \
{ \
	if (__builtin_expect(_nanos6_lib_handle == NULL, 0)) { \
		fprintf(stderr, "Nanos 6 loader error: call to " #fname " before library initialization\n"); \
		abort(); \
	} \
	\
	void *symbol = (void (*)(void)) dlsym(_nanos6_lib_handle, #fname); \
	if ((symbol == NULL) && (#fallback != NULL)) { \
		symbol = (void (*)(void)) fallback; \
	} \
	if (symbol == NULL) { \
		fprintf(stderr, "Nanos 6 loader error: " #area " runtime function " #fname " is undefined in '%s'\n", _nanos6_lib_filename); \
		abort(); \
	} \
	\
	return (void (*)(void)) symbol; \
} \
\
void *fname() __attribute__ (( ifunc("_" #fname "_resolver") ))


#endif /* RESOLVE_H */
