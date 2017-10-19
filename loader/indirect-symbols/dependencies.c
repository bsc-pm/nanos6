/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void nanos_register_read_depinfo(void *handler, void *start, size_t length)
{
	typedef void nanos_register_read_depinfo_t(void *handler, void *start, size_t length);
	
	static nanos_register_read_depinfo_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_register_read_depinfo_t *) _nanos6_resolve_symbol("nanos_register_read_depinfo", "dependency", NULL);
	}
	
	(*symbol)(handler, start, length);
}


void nanos_register_write_depinfo(void *handler, void *start, size_t length)
{
	typedef void nanos_register_write_depinfo_t(void *handler, void *start, size_t length);
	
	static nanos_register_write_depinfo_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_register_write_depinfo_t *) _nanos6_resolve_symbol("nanos_register_write_depinfo", "dependency", NULL);
	}
	
	(*symbol)(handler, start, length);
}


void nanos_register_readwrite_depinfo(void *handler, void *start, size_t length)
{
	typedef void nanos_register_readwrite_depinfo_t(void *handler, void *start, size_t length);
	
	static nanos_register_readwrite_depinfo_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_register_readwrite_depinfo_t *) _nanos6_resolve_symbol("nanos_register_readwrite_depinfo", "dependency", NULL);
	}
	
	(*symbol)(handler, start, length);
}


void nanos_register_commutative_depinfo(void *handler, void *start, size_t length)
{
	typedef void nanos_register_commutative_depinfo_t(void *handler, void *start, size_t length);
	
	static nanos_register_commutative_depinfo_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_register_commutative_depinfo_t *) _nanos6_resolve_symbol("nanos_register_commutative_depinfo", "commutative dependency", "nanos_register_readwrite_depinfo");
	}
	
	(*symbol)(handler, start, length);
}


void nanos_register_concurrent_depinfo(void *handler, void *start, size_t length)
{
	typedef void nanos_register_concurrent_depinfo_t(void *handler, void *start, size_t length);
	
	static nanos_register_concurrent_depinfo_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_register_concurrent_depinfo_t *) _nanos6_resolve_symbol("nanos_register_concurrent_depinfo", "concurrent dependency", "nanos_register_readwrite_depinfo");
	}
	
	(*symbol)(handler, start, length);
}


#pragma GCC visibility pop
