/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_STREAM_HPP
#define CUDA_STREAM_HPP

#include <cuda_runtime_api.h>

#include "lowlevel/cuda/CUDAErrorHandler.hpp"

class CUDAComputePlace;

class CUDAStream {

private:
	size_t _index;
	cudaStream_t _stream;
	
public:
	CUDAStream(size_t index): _index(index)
	{
		cudaError_t err = cudaStreamCreate(&_stream);
		CUDAErrorHandler::handle(err, "When creating stream");
	}
	
	//Disable copy constructor
	CUDAStream(CUDAStream const &) = delete;
	CUDAStream operator=(CUDAStream const &) = delete;	

	~CUDAStream()
	{
		cudaError_t err = cudaStreamDestroy(_stream);
		CUDAErrorHandler::handle(err, "When destroying stream");
	}

	//! \brief Get the assigned index of the stream
	size_t getIndex() const 
	{
		return _index;
	}
	
	//! \brief Get the underlying cudaStream_t object
	cudaStream_t getStream() const
	{
		return _stream;
	}
};

#endif
