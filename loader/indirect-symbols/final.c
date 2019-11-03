/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

static signed int signed_int_always_false(void) { return 0; }
signed int nanos6_in_final(void)
{
	typedef signed int nanos6_in_final_t(void);

	static nanos6_in_final_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_in_final_t *) _nanos6_resolve_symbol_with_local_fallback("nanos6_in_final", "final tasks", signed_int_always_false, "always false");
	}

	return (*symbol)();
}

signed int nanos6_in_serial_context(void)
{
	typedef signed int nanos6_in_serial_context_t(void);

	static nanos6_in_serial_context_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_in_serial_context_t *) _nanos6_resolve_symbol_with_local_fallback("nanos6_in_serial_context", "final tasks", signed_int_always_false, "always false");
	}

	return (*symbol)();
}


#pragma GCC visibility pop
