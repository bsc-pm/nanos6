/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "resolve.h"


#pragma GCC visibility push(default)

void nanos6_register_polling_service(char const *service_name, nanos6_polling_service_t service_function, void *service_data)
{
	typedef void nanos6_register_polling_service_t(char const *service_name, nanos6_polling_service_t service_function, void *service_data);
	
	static nanos6_register_polling_service_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_register_polling_service_t *) _nanos6_resolve_symbol("nanos6_register_polling_service", "polling services", NULL);
	}
	
	(*symbol)(service_name, service_function, service_data);
}


void nanos6_unregister_polling_service(char const *service_name, nanos6_polling_service_t service_function, void *service_data)
{
	typedef void nanos6_unregister_polling_service_t(char const *service_name, nanos6_polling_service_t service_function, void *service_data);
	
	static nanos6_unregister_polling_service_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_unregister_polling_service_t *) _nanos6_resolve_symbol("nanos6_unregister_polling_service", "polling services", NULL);
	}
	
	(*symbol)(service_name, service_function, service_data);
}

#pragma GCC visibility pop

