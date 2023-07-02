/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(f...)                                                    \
    {                                                                       \
        const cudaError_t __err = f;                                        \
        if (__err != cudaSuccess) {                                         \
            printf("Error: '%s' [%s:%i]: %i\n",#f,__FILE__,__LINE__,__err); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

#endif // CUDA_UTILS_HPP
