#include "resolve.h"


void *nanos_get_original_reduction_address(const void *private_address)
{
	typedef void *nanos_get_original_reduction_address(const void *private_address);
	
	static nanos_get_original_reduction_address_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_get_original_reduction_address_t *)
			_nanos6_resolve_symbol("nanos_get_original_reduction_address",
					"reductions", NULL);
	}
	
	(*symbol)(private_address);
}
