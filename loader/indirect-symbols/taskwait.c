#include "resolve.h"


void nanos_taskwait(char const *invocation_source)
{
	typedef void nanos_taskwait_t(char const *invocation_source);
	
	static nanos_taskwait_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_taskwait_t *) _nanos6_resolve_symbol("nanos_taskwait", "essential", NULL);
	}
	
	(*symbol)(invocation_source);
}


