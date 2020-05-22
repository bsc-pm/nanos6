/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_FUNCTIONS_HPP
#define CUDA_FUNCTIONS_HPP

#include <cuda_runtime_api.h>

#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/cuda/CUDAErrorHandler.hpp"

// A helper class, providing static helper functions, specific to the device,
// to be used by DeviceInfo and other relevant classes as utilities.
class CUDAFunctions {

public:
	static bool initialize()
	{
		// Dummy setDevice operation to initialize CUDA runtime;
		// if even 1 GPU is present setting to 0 should always
		// be succesful.
		cudaError_t err = cudaSetDevice(0);
		CUDAErrorHandler::handle(err, "During CUDA initialization");
		return err == cudaSuccess;
	}


	static size_t getDeviceCount()
	{
		int deviceCount = 0;
		cudaError_t err = cudaGetDeviceCount(&deviceCount);
		CUDAErrorHandler::handle(err, "When counting devices");
		if (err != cudaSuccess)
			return 0;
		return (size_t)deviceCount;
	}

	static void getDeviceProperties(cudaDeviceProp &deviceProp, int device)
	{
		CUDAErrorHandler::handle(cudaGetDeviceProperties(&deviceProp, device), "While getting CUDA device properties");
	}

	static size_t getPageSize()
	{
		static EnvironmentVariable<size_t> pageSize("NANOS6_CUDA_PAGESIZE", 0x8000);
		return pageSize;
	}


	static void setActiveDevice(int device)
	{
		CUDAErrorHandler::handle(cudaSetDevice(device), "While setting CUDA device");
	}

	static cudaStream_t createStream()
	{
		cudaStream_t stream;
		CUDAErrorHandler::handle(cudaStreamCreate(&stream), "While creating CUDA stream");
		return stream;
	}

	static void destroyStream(cudaStream_t &stream)
	{
		CUDAErrorHandler::handle(cudaStreamDestroy(stream), "While destroying CUDA stream");
	}

	static void *malloc(size_t size)
	{
		void *ptr;
		cudaError_t err = cudaMalloc(&ptr, size);
		CUDAErrorHandler::handle(err, "In device malloc");
		if (err != cudaSuccess)
			return nullptr;
		return ptr;
	}

	static void createEvent(cudaEvent_t &event)
	{
		CUDAErrorHandler::handle(cudaEventCreate(&event), "While creating CUDA event");
	}

	static void destroyEvent(cudaEvent_t &event)
	{
		CUDAErrorHandler::handle(cudaEventDestroy(event), "While destroying CUDA event");
	}

	static void recordEvent(cudaEvent_t &event, cudaStream_t &stream)
	{
		CUDAErrorHandler::handle(cudaEventRecord(event, stream), "While recording CUDA event");
	}

	static bool cudaEventFinished(cudaEvent_t &event)
	{
		return (cudaEventQuery(event) == cudaSuccess);
	}

};

#endif // CUDA_FUNCTIONS_HPP
