#include "resolve.h"


void nanos_register_polling_service(char const *service_name, nanos_polling_service_t service_function, void *service_data)
{
	typedef void nanos_register_polling_service_t(char const *service_name, nanos_polling_service_t service_function, void *service_data);
	
	static nanos_register_polling_service_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_register_polling_service_t *) _nanos6_resolve_symbol("nanos_register_polling_service", "polling services", NULL);
	}
	
	(*symbol)(service_name, service_function, service_data);
}


void nanos_unregister_polling_service(char const *service_name, nanos_polling_service_t service_function, void *service_data)
{
	typedef void nanos_unregister_polling_service_t(char const *service_name, nanos_polling_service_t service_function, void *service_data);
	
	static nanos_unregister_polling_service_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos_unregister_polling_service_t *) _nanos6_resolve_symbol("nanos_unregister_polling_service", "polling services", NULL);
	}
	
	(*symbol)(service_name, service_function, service_data);
}

