/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void nanos6_user_lock(void **handlerPointer, char const *invocation_source)
{
	typedef void nanos6_user_lock_t(void **handlerPointer, char const *invocation_source);
	
	static nanos6_user_lock_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_user_lock_t *) _nanos6_resolve_symbol("nanos6_user_lock", "user-side mutex", NULL);
	}
	
	(*symbol)(handlerPointer, invocation_source);
}


void nanos6_user_unlock(void **handlerPointer)
{
	typedef void nanos6_user_unlock_t(void **handlerPointer);
	
	static nanos6_user_unlock_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_user_unlock_t *) _nanos6_resolve_symbol("nanos6_user_unlock", "user-side mutex", NULL);
	}
	
	(*symbol)(handlerPointer);
}

#pragma GCC visibility pop

