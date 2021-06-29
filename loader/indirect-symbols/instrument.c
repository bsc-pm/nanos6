/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"

#include <stdint.h>


#pragma GCC visibility push(default)

int nanos6_is_distributed_instrument_enabled(void)
{
	typedef int nanos6_is_distributed_instrument_enabled_t(void);

	static nanos6_is_distributed_instrument_enabled_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_is_distributed_instrument_enabled_t *) _nanos6_resolve_symbol("nanos6_is_distributed_instrument_enabled", "instrument", NULL);
	}

	return (*symbol)();
}

void nanos6_setup_distributed_instrument(const nanos6_distributed_instrument_info_t *info)
{
	typedef void nanos6_setup_distributed_instrument_t(const nanos6_distributed_instrument_info_t *info);

	static nanos6_setup_distributed_instrument_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_setup_distributed_instrument_t *) _nanos6_resolve_symbol("nanos6_setup_distributed_instrument", "instrument", NULL);
	}

	(*symbol)(info);
}

int64_t nanos6_get_instrument_start_time_ns(void)
{
	typedef int64_t nanos6_get_instrument_start_time_ns_t(void);

	static nanos6_get_instrument_start_time_ns_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_instrument_start_time_ns_t *) _nanos6_resolve_symbol("nanos6_get_instrument_start_time_ns", "instrument", NULL);
	}

	return (*symbol)();
}

#pragma GCC visibility pop

