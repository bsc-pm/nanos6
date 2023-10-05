/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_FUNCTIONS_HPP
#define CUDA_FUNCTIONS_HPP

#include <cuda_runtime_api.h>

#include "CUDARuntimeLoader.hpp"
#include "lowlevel/cuda/CUDAErrorHandler.hpp"
#include "support/config/ConfigVariable.hpp"


// A helper class, providing static helper functions, specific to the device,
// to be used by DeviceInfo and other relevant classes as utilities.
class CUDAFunctions {
	static CUDARuntimeLoader &getCudaRuntimeLoader()
	{
		static CUDARuntimeLoader loader;
		return loader;
	}

public:
	static bool initialize()
	{
		static bool initialized = false;
		static bool initializedStatus = false;
		if (initialized)
			return initializedStatus;

		initialized = true;

		CUresult st = cuInit(0);
		if (st != CUDA_SUCCESS) {
			initializedStatus = false;
			return initializedStatus;
		}

		// Try to load all kernels
		getCudaRuntimeLoader();

		initializedStatus = true;
		return initializedStatus;
	}

	static CUfunction loadFunction(const char *str)
	{
		return getCudaRuntimeLoader().loadFunction(str);
	}

	static size_t getDeviceCount()
	{
		int deviceCount = 0;
		cudaError_t err = cudaGetDeviceCount(&deviceCount);
		if (err != cudaSuccess) {
			if (err != cudaErrorNoDevice) {
				CUDAErrorHandler::warn(err, " received during CUDA device detection. ",
					"Nanos6 was compiled with CUDA support but the driver returned error.",
					"\nRunning CUDA tasks is disabled");
			}
			return 0;
		}
		return (size_t)deviceCount;
	}

	static void getDeviceProperties(cudaDeviceProp &deviceProp, int device)
	{
		CUDAErrorHandler::handle(cudaGetDeviceProperties(&deviceProp, device),
			"While getting CUDA device properties");
	}

	static size_t getPageSize()
	{
		static ConfigVariable<size_t> pageSize("devices.cuda.page_size");
		return pageSize;
	}

	static int getActiveDevice()
	{
		int device;
		CUDAErrorHandler::handle(cudaGetDevice(&device), "While getting CUDA device");
		return device;
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
		CUDAErrorHandler::warn(cudaStreamDestroy(stream), "While destroying CUDA stream");
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

	// Allocate special pinned memory that allows sped up transfers between host and GPU.
	static void *mallocHost(size_t size)
	{
		void *ptr;
		cudaError_t err = cudaMallocHost(&ptr, size);
		CUDAErrorHandler::handle(err, "Allocating CUDA pinned host memory");
		if (err != cudaSuccess)
			return nullptr;
		return ptr;
	}

	static void free(void *ptr)
	{
		cudaError_t err = cudaFree(ptr);
		CUDAErrorHandler::handle(err, "Freeing device memory");
	}

	static void freeHost(void *ptr)
	{
		cudaError_t err = cudaFreeHost(ptr);
		CUDAErrorHandler::handle(err, "Freeing pinned host memory");
	}

	static void createEvent(cudaEvent_t &event)
	{
		CUDAErrorHandler::handle(cudaEventCreate(&event), "While creating CUDA event");
	}

	static void destroyEvent(cudaEvent_t &event)
	{
		CUDAErrorHandler::warn(cudaEventDestroy(event), "While destroying CUDA event");
	}

	static void recordEvent(cudaEvent_t &event, cudaStream_t &stream)
	{
		CUDAErrorHandler::handle(cudaEventRecord(event, stream), "While recording CUDA event");
	}

	static bool isEventFinished(cudaEvent_t &event)
	{
		return CUDAErrorHandler::handleEvent(
			cudaEventQuery(event), "While querying event");
	}

	static void prefetchMemory(void *pHost, size_t size, int device, cudaStream_t &stream, bool readOnly)
	{
		// Ensure that we have a stream assigned; stream 0 is special in CUDA and we do not
		// launch tasks are never launched on it
		assert(stream != 0);

		if (size == 0)
			return;

		// Depending on the access we're prefetching, we will advise the driver to do a shared copy
		cudaMemoryAdvise advice = (readOnly ? cudaMemAdviseSetReadMostly : cudaMemAdviseUnsetReadMostly);
		cudaError_t err = cudaMemAdvise(pHost, size, advice, device);

		// The memory may not be managed memory, and thus, skip prefetching
		if (err == cudaErrorInvalidValue)
			return;

		// Check the rest of errors
		CUDAErrorHandler::handle(err, "Advising memory region");

		// Call a prefetch operation on the same stream that we are going to launch that task on
		err = cudaMemPrefetchAsync(pHost, size, device, stream);
		CUDAErrorHandler::handle(err, "Prefetching memory to device");
	}

	static void copyMemory(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
	{
		cudaError_t err = cudaMemcpy(dst, src, count, kind);
		CUDAErrorHandler::handle(err, "Copying memory");
	}

	static void copyMemoryAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
	{
		cudaError_t err = cudaMemcpyAsync(dst, src, count, kind, stream);
		CUDAErrorHandler::handle(err, "Copying memory");
	}

	static void copyMemoryP2PAsync(void *dstAddress, int dstDevice, const void *srcAddress, int srcDevice, size_t size, cudaStream_t stream)
	{
		cudaError_t err = cudaMemcpyPeerAsync(dstAddress, dstDevice, srcAddress, srcDevice, size, stream);
		CUDAErrorHandler::handle(err, "Copying memory P2P");
	}

	static void waitForEvent(cudaEvent_t event, cudaStream_t stream)
	{
		cudaError_t err = cudaStreamWaitEvent(stream, event, 0);
		CUDAErrorHandler::handle(err, "Waiting for an event in a different stream");
	}

	static void launchKernel(
		const char *kernelName, void **kernelParams,
		size_t gridDim1, size_t gridDim2, size_t gridDim3,
		size_t blockDim1, size_t blockDim2, size_t blockDim3,
		size_t sharedMemoryBytes, CUstream stream
	) {
		CUresult res = cuLaunchKernel(loadFunction(kernelName),
			gridDim1, gridDim2, gridDim3, blockDim1, blockDim2, blockDim3,
			sharedMemoryBytes, stream, kernelParams, nullptr);

		if (res != CUDA_SUCCESS) {
			const char *errorDescription;
			res = cuGetErrorString(res, &errorDescription);
			if (res != CUDA_SUCCESS)
				errorDescription = "Unknown error";

			FatalErrorHandler::fail(
				"Failed when launching kernel ", kernelName, ":\n"
				"    error: ", errorDescription, "\n"
				"    configuration:\n"
				"        grid: ", gridDim1, " x ", gridDim2, " x ", gridDim3, "\n"
				"        block: ", blockDim1, " x ", blockDim2, " x ", blockDim3, "\n"
				"        shmem: ", sharedMemoryBytes, " bytes");
		}
	}
};

#endif // CUDA_FUNCTIONS_HPP
