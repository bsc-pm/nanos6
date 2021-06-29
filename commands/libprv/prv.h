/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef LIBPRV_PRV_H
#define LIBPRV_PRV_H

#include <stdint.h>
#include "uthash.h"

#define MAX_LABEL 256
#define MAX_SRCLINE 256

struct task_type {
	uint64_t type;
	char label[MAX_LABEL];
	char srcline[MAX_SRCLINE];
	UT_hash_handle hh;
};

#endif // LIBPRV_PRV_H
