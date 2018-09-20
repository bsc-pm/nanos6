/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_POLLING_H
#define NANOS6_POLLING_H

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_polling_api
enum nanos6_polling_api_t { nanos6_polling_api = 1 };


#ifdef __cplusplus
extern "C" {
#endif


//! \brief Function that the runtime calls periodically
//! 
//! \param service_data a pointer to data that the function uses and that
//! also identifies an instance of this service registration
//! 
//! \returns true to signal that the purpose of the function has been
//! achieved and that the function should not be called again with the given
//! service_data
typedef int (*nanos6_polling_service_t)(void *service_data);


//! \brief Register a function and parameter of that function that the runtime
//! must call periodically to perform operations in a non-blocking fashion
//! 
//! Registers a function that the runtime will called periodically. The  purpose
//! is to support operations in a non-blocking way. For instance to check for
//! certain events that are not possible to check in a blocking way, or that it
//! is not desirable to do so.
//! 
//! The service_data parameter is an opaque pointer that is passed to the function
//! as is.
//! 
//! The same function can be registered several times with different service_data
//! parameters, and each combination will be considered as a different
//! registration. That is, a registered service instance consists of a service
//! function and a specific service_data.
//! 
//! The service will remain registered as long as the function returns false and
//! the service has not been explicitly deregistered through a call to
//! nanos6_unregister_polling_service.
//! 
//! \param[in] service_name a string that identifies the kind of service that will
//! be serviced
//! \param[in] service the function that the runtime should call periodically
//! \param service_data an opaque pointer to data that is passed to the service
//! function
void nanos6_register_polling_service(char const *service_name, nanos6_polling_service_t service_function, void *service_data);


//! \brief Unregister a function and parameter of that function previously
//! registered through a call to nanos6_register_polling_service.
//! 
//! Unregister a service instance identified by the service_function and
//! service_data previously registered through a call to
//! nanos6_register_polling_service. The service_function must not have returned
//! true when called with the given service_data, since that return value is
//! equivalent to a call to this function.
//! 
//! \param[in] service_name a string that identifies the kind of service
//! \param[in] service the function that the runtime should stop calling
//! periodically with the given service_data
//! \param service_data an opaque pointer to the data that was passed to the
//! service function
void nanos6_unregister_polling_service(char const *service_name, nanos6_polling_service_t service_function, void *service_data);


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_POLLING_H */
