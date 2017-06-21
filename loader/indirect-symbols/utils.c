#include "resolve.h"


void nanos6_bzero(void *buffer, size_t size)
{
	typedef void nanos6_bzero_t(void *buffer, size_t size);
	
	static nanos6_bzero_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_bzero_t *) _nanos6_resolve_symbol("nanos6_bzero", "auxiliary functionality", NULL);
	}
	
	(*symbol)(buffer, size);
}
