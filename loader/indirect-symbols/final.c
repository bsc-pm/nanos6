#include "resolve.h"


static signed int signed_int_always_false(void) { return 0; }
signed int nanos_in_final(void)
{
	typedef signed int nanos_in_final_t();
	
	static nanos_in_final_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_in_final_t *) _nanos6_resolve_symbol_with_local_fallback("nanos_in_final", "final tasks", signed_int_always_false, "always false");
	}
	
	return (*symbol)();
}

