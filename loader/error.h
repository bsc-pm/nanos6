#ifndef NANOS6_LOADER_ERROR_H
#define NANOS6_LOADER_ERROR_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdlib.h>

#include "loader.h"


#define handle_error() \
	do { \
		if (_nanos6_has_started) { \
			abort(); \
		} else { \
			_nanos6_exit_with_error = 1; \
		} \
	} while (0)


#endif /* NANOS6_LOADER_ERROR_H */
	
