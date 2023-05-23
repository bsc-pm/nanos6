/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_VERSION_H
#define NANOS6_VERSION_H

#include <stdint.h>

#pragma GCC visibility push(default)

#ifdef __cplusplus
extern "C" {
#endif

//! \brief A version is composed by its family, major and minor numbers
typedef struct
{
	uint64_t family;
	uint64_t major_version;
	uint64_t minor_version;
} nanos6_version_t;

//! \brief Check whether the runtime is compatible with required versions
//!
//! The compiler can issue calls to this function passing the versions that the
//! user program is relying on. The runtime should be compatible with these
//! versions. Otherwise, it should report an error and abort the execution
//!
//! \param[in] size the number of version families to be checked
//! \param[in] versions the version families to be checked
//! \param[in] source a string with the source file where the check comes from
void nanos6_check_version(uint64_t size, nanos6_version_t *versions, const char *source);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_VERSION_H */
