/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_USER_MUTEX_H
#define NANOS6_USER_MUTEX_H

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_locking_api
enum nanos6_locking_api_t { nanos6_locking_api = 1 };


#ifdef __cplusplus
extern "C" {
#endif


//! \brief User-side lock primitive with initialization on first call
//!
//! Performs an user-side lock over a mutex (of type void *) that must be initially
//! initialized to nullptr. The first call to this function performs the actual
//! mutex allocation and stores the handler in the address that is passed.
//!
//! \param[in,out] handlerPointer a pointer to the handler, which is of type void *, that represent the mutex
//! \param[in] invocation_source A string that identifies the location of the critical region in the source code
void nanos6_user_lock(void **handlerPointer, char const *invocation_source);

//! \brief User-side unlock primitive
//!
//! Performs an user-side unlock over a mutex (of type void *) initialized during
//! the first call to nanos6_user_lock.
//!
//! \param[in] handlerPointer a pointer to the handler, which is of type void *, that represent the mutex
void nanos6_user_unlock(void **handlerPointer);


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_USER_MUTEX_H */
