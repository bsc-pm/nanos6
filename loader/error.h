/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef NANOS6_LOADER_ERROR_H
#define NANOS6_LOADER_ERROR_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>

#include "loader.h"

// Cannot go back from this.
#define handle_error() \
	do { \
		abort(); \
	} while (0)


#endif /* NANOS6_LOADER_ERROR_H */

