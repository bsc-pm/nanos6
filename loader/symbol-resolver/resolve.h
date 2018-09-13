/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef RESOLVE_H
#define RESOLVE_H


#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "error.h"
#include "loader.h"

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>


char const *nanos6_get_runtime_path(void);
void _nanos6_loader(void);


#define RESOLVE_API_FUNCTION(fname, area, fallback) \
void (*_##fname##_resolver(void)) (void) \
{ \
	if (__builtin_expect(_nanos6_lib_handle == NULL, 0)) { \
		fprintf(stderr, "Nanos 6 loader warning: attempt to resolve the address of " #fname " before library initialization\n"); \
		_nanos6_loader(); \
	} \
	\
	if (__builtin_expect(_nanos6_lib_handle == NULL, 0)) { \
		handle_error(); \
		return NULL; \
	} \
	\
	void *symbol = (void (*)(void)) dlsym(_nanos6_lib_handle, #fname); \
	if ((symbol == NULL) && (#fallback != NULL)) { \
		symbol = (void (*)(void)) dlsym(_nanos6_lib_handle, #fallback); \
		if (symbol != NULL) { \
			fprintf(stderr, "Nanos 6 loader warning: " #area " runtime function " #fname " is undefined in '%s' falling back to function " #fallback " instead\n", nanos6_get_runtime_path()); \
		} \
	} \
	if (symbol == NULL) { \
		fprintf(stderr, "Nanos 6 loader error: " #area " runtime function " #fname " is undefined in '%s'\n", nanos6_get_runtime_path()); \
		handle_error(); \
		return NULL; \
	} \
	\
	return (void (*)(void)) symbol; \
} \
void *fname() __attribute__ (( ifunc("_" #fname "_resolver"), visibility("default") ))


#define RESOLVE_API_FUNCTION_WITH_LOCAL_FALLBACK(fname, area, fallback) \
void (*_##fname##_resolver(void)) (void) \
{ \
	if (__builtin_expect(_nanos6_lib_handle == NULL, 0)) { \
		fprintf(stderr, "Nanos 6 loader error: call to " #fname " before library initialization\n"); \
		handle_error(); \
		return NULL; \
	} \
	\
	void *symbol = (void (*)(void)) dlsym(_nanos6_lib_handle, #fname); \
	if ((symbol == NULL) && (#fallback != NULL)) { \
		symbol = (void (*)(void)) fallback; \
	} \
	if (symbol == NULL) { \
		fprintf(stderr, "Nanos 6 loader error: " #area " runtime function " #fname " is undefined in '%s'\n", nanos6_get_runtime_path()); \
		handle_error(); \
		return NULL; \
	} \
	\
	return (void (*)(void)) symbol; \
} \
\
void *fname() __attribute__ (( ifunc("_" #fname "_resolver"), visibility("default") ))


#define RESOLVE_INTERCEPTED_FUNCTION_WITH_GLOBAL_FALLBACK(fname, area, rtype, ...) \
void (*_##fname##_resolver(void)) (void) \
{ \
	if (__builtin_expect(_nanos6_lib_handle == NULL, 0)) { \
		_nanos6_loader(); \
		if (_nanos6_lib_handle == NULL) { \
			fprintf(stderr, "Nanos 6 loader error: call to " #fname " before library initialization\n"); \
			handle_error(); \
		return NULL; \
		} \
	} \
	\
	void *symbol = (void (*)(void)) dlsym(_nanos6_lib_handle, "nanos6_intercepted_" #fname); \
	if ((symbol == NULL)) { \
		symbol = (void (*)(void)) dlsym(RTLD_NEXT, #fname); \
	} \
	if (symbol == NULL) { \
		fprintf(stderr, "Nanos 6 loader error: " #area " runtime function " #fname " is undefined in '%s'\n", nanos6_get_runtime_path()); \
		handle_error(); \
		return NULL; \
	} \
	\
	return (void (*)(void)) symbol; \
} \
\
rtype fname(__VA_ARGS__) __attribute__ (( ifunc("_" #fname "_resolver"), visibility("default") ))


#endif /* RESOLVE_H */
