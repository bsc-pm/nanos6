/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2019 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_SEL_HPP
#define CUDA_SEL_HPP

#include <config.h>

#ifdef USE_CUDA
#include "cuda/CUDAFunctions.hpp"
#else
#include "NULLDevice.hpp"
typedef NULLDevice CUDAFunctions;
#endif

#endif
