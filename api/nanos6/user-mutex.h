#ifndef NANOS6_USER_MUTEX_H
#define NANOS6_USER_MUTEX_H

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
void nanos_user_lock(void **handlerPointer, char const *invocation_source);

//! \brief User-side unlock primitive
//!
//! Performs an user-side unlock over a mutex (of type void *) initialized during
//! the first call to nanos_user_lock.
//!
//! \param[in] handlerPointer a pointer to the handler, which is of type void *, that represent the mutex
void nanos_user_unlock(void **handlerPointer);


#ifdef __cplusplus
}
#endif


#endif /* NANOS6_USER_MUTEX_H */
