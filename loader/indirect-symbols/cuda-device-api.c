/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "resolve.h"

#pragma GCC visibility push(default)

cudaStream_t nanos6_get_current_cuda_stream(void)
{
	typedef cudaStream_t nanos6_get_current_cuda_stream_t(void);

	static nanos6_get_current_cuda_stream_t *symbol = NULL;
	if (__builtin_expect(symbol == NULL, 0)) {
		symbol = (nanos6_get_current_cuda_stream *)
			_nanos6_resolve_symbol("nanos6_get_current_cuda_stream", "cuda", NULL);
	}

	return (*symbol)();
}

#pragma GCC visibility pop
