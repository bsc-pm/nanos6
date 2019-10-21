/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_LINT_H
#define NANOS6_LINT_H

#include "major.h"


#pragma GCC visibility push(default)


// NOTE: The full version depends also on nanos6_major_api
//       That is:   nanos6_major_api . nanos6_lint_api
enum nanos6_lint_api_t { nanos6_lint_api = 1 };

#ifdef __cplusplus
extern "C" {
#endif



void nanos6_lint_ignore_region_begin(void);

void nanos6_lint_ignore_region_end(void);

void nanos6_lint_register_alloc(
	void *base_address,
	unsigned long size
);

void nanos6_lint_register_free(
	void *base_address
);



#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif /* NANOS6_LINT_H */
