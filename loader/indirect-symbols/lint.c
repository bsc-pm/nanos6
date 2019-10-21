/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)


void nanos6_lint_ignore_region_begin()
{
	typedef void *nanos6_lint_ignore_region_begin_t();
	
	static nanos6_lint_ignore_region_begin_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_lint_ignore_region_begin_t *) _nanos6_resolve_symbol(
			"nanos6_lint_ignore_region_begin", "lint", NULL);
	}
	
	(*symbol)();
}


void nanos6_lint_ignore_region_end()
{
	typedef void nanos6_lint_ignore_region_end_t();
	
	static nanos6_lint_ignore_region_end_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_lint_ignore_region_end_t *) _nanos6_resolve_symbol(
			"nanos6_lint_ignore_region_end", "lint", NULL);
	}
	
	(*symbol)();
}


void nanos6_lint_register_alloc(void *base_address, size_t size)
{
	typedef void nanos6_lint_register_alloc_t(void *base_address, size_t size);
	
	static nanos6_lint_register_alloc_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_lint_register_alloc_t *) _nanos6_resolve_symbol(
			"nanos6_lint_register_alloc", "lint", NULL);
	}
	
	(*symbol)(base_address, size);
}


void nanos6_lint_register_free(void *base_address)
{
	typedef void nanos6_lint_register_free_t(void *base_address);
	
	static nanos6_lint_register_free_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_lint_register_free_t *) _nanos6_resolve_symbol(
			"nanos6_lint_register_free", "lint", NULL);
	}
	
	(*symbol)(base_address);
}


#pragma GCC visibility pop
