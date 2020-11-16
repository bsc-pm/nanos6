/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_CONFIG_H
#define NANOS6_CONFIG_H

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_config_api
enum nanos6_config_api_t { nanos6_config_api = 1 };


#ifdef __cplusplus
extern "C" {
#endif

void nanos6_config_assert(const char *config_condition);

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop


#endif // NANOS6_CONFIG_H
