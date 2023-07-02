/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include <nanos6/debug.h>

#define UNUSED __attribute__((unused))

#define CHECK(f...)                                                         \
    {                                                                       \
        const int __r = f;                                                  \
        if (__r) {                                                          \
            printf("Error: '%s' [%s:%i]: %i\n",#f,__FILE__,__LINE__,__r);   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

namespace Utils {

static inline void warmup()
{
	const size_t ncpus = nanos6_get_num_cpus();
	size_t counter = 0;

	for (size_t t = 0; t < ncpus; ++t) {
		#pragma oss task shared(counter)
		{
			#pragma oss atomic
			counter++;

			while (counter < ncpus) {
				usleep(100);
				__sync_synchronize();
			}
		}
	}
	#pragma oss taskwait
}

} // namespace Utils

#endif // UTILS_HPP
