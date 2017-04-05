#include "resolve.h"


void nanos_user_lock(void **handlerPointer, char const *invocation_source)
{
	typedef void nanos_user_lock_t(void **handlerPointer, char const *invocation_source);
	
	static nanos_user_lock_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_user_lock_t *) _nanos6_resolve_symbol("nanos_user_lock", "user-side mutex", NULL);
	}
	
	(*symbol)(handlerPointer, invocation_source);
}


void nanos_user_unlock(void **handlerPointer)
{
	typedef void nanos_user_unlock_t(void **handlerPointer);
	
	static nanos_user_unlock_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_user_unlock_t *) _nanos6_resolve_symbol("nanos_user_unlock", "user-side mutex", NULL);
	}
	
	(*symbol)(handlerPointer);
}


