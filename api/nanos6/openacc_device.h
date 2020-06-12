/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_OPENACC_DEVICE_H
#define NANOS6_OPENACC_DEVICE_H

#include "major.h"

#pragma GCC visibility push(default)

// NOTE: The full version depends also on nanos6_major_api
// That is:   nanos6_major_api . nanos6_openacc_device_api
enum nanos6_openacc_device_api_t { nanos6_openacc_device_api = 1 };

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	int asyncId;
} nanos6_openacc_device_environment_t;

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif /* NANOS6_OPENACC_DEVICE_H */

