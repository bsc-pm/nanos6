#ifndef NANOS6_DEPENDENCIES_H
#define NANOS6_DEPENDENCIES_H

#include <stddef.h>

#pragma GCC visibility push(default)

enum nanos6_unidimensional_dependencies_api_t { nanos6_unidimensional_dependencies_api = 1 };


#ifdef __cplusplus
extern "C" {
#endif


//! \brief Register a task read access on linear region of addresses
//!
//! \param[in] handler the handler received in register_depinfo
//! \param[in] start first address accessed
//! \param[in] length number of bytes until and including the last byte accessed
void nanos_register_read_depinfo(void *handler, void *start, size_t length);

//! \brief Register a task write access on linear region of addresses
//!
//! \param[in] handler the handler received in register_depinfo
//! \param[in] start first address accessed
//! \param[in] length number of bytes until and including the last byte accessed
void nanos_register_write_depinfo(void *handler, void *start, size_t length);

//! \brief Register a task read and write access on linear region of addresses
//!
//! \param[in] handler the handler received in register_depinfo
//! \param[in] start first address accessed
//! \param[in] length number of bytes until and including the last byte accessed
void nanos_register_readwrite_depinfo(void *handler, void *start, size_t length);

//! \brief Register a task commutative access on linear region of addresses
//!
//! \param[in] handler the handler received in register_depinfo
//! \param[in] start first address accessed
//! \param[in] length number of bytes until and including the last byte accessed
void nanos_register_commutative_depinfo(void *handler, void *start, size_t length);

//! \brief Register a task concurrent access on linear region of addresses
//!
//! \param[in] handler the handler received in register_depinfo
//! \param[in] start first address accessed
//! \param[in] length number of bytes until and including the last byte accessed
void nanos_register_concurrent_depinfo(void *handler, void *start, size_t length);


//! \brief Register a weak task read access on linear region of addresses
//!
//! \param[in] handler the handler received in register_depinfo
//! \param[in] start first address accessed
//! \param[in] length number of bytes until and including the last byte accessed
void nanos_register_weak_read_depinfo(void *handler, void *start, size_t length);

//! \brief Register a weak task write access on linear region of addresses
//!
//! \param[in] handler the handler received in register_depinfo
//! \param[in] start first address accessed
//! \param[in] length number of bytes until and including the last byte accessed
void nanos_register_weak_write_depinfo(void *handler, void *start, size_t length);

//! \brief Register a weak task read and write access on linear region of addresses
//!
//! \param[in] handler the handler received in register_depinfo
//! \param[in] start first address accessed
//! \param[in] length number of bytes until and including the last byte accessed
void nanos_register_weak_readwrite_depinfo(void *handler, void *start, size_t length);


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_DEPENDENCIES_H */
