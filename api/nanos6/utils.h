/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_UTILS_H
#define NANOS6_UTILS_H

#include "major.h"

#include <stddef.h>


#pragma GCC visibility push(default)

enum nanos6_utils_api_t { nanos6_utils_api = 1 };

#ifdef __cplusplus
extern "C" {
#endif



// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_utils_api
//! \brief Fill up a buffer with zeros
void nanos6_bzero(void *buffer, size_t size);


#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif /* NANOS6_UTILS_H */
