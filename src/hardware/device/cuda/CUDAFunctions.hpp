/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
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
	static std::vector<CUdevice> &getCudaDevices(int num)
	{
		static std::vector<CUdevice> cdvs(num);
		return cdvs;
	}
	static std::vector<CUcontext> &getCudaPrimaryContexts(int num)
	{
		static std::vector<CUcontext> pctx(num);
		return pctx;
	}

	static CUDARuntimeLoader &getCudaRuntimeLoader()
	{
		static CUDARuntimeLoader loader(getCudaPrimaryContexts());
		return loader;
	}

public:
	static bool initialize()
	{
		static bool initialized = false;
		static bool initializedStatus = false;
		if (initialized)
			return initializedStatus;

		CUresult st = cuInit(0);
		if (st != CUDA_SUCCESS) {
			return initializedStatus;
		}

		int devNum = getDeviceCount();
		auto &cDevices = getCudaDevices(devNum);
		auto &cPrimaryCtx = getCudaPrimaryContexts(devNum);
		// Initialize the primary context, this context is special and shared with the runtime api
		for (int i = 0; i < devNum; ++i) {
			FatalErrorHandler::failIf(cuDeviceGet(&cDevices[i], i) != CUDA_SUCCESS,
				"Failed to get device " + std::to_string(i));
			FatalErrorHandler::failIf(cuDevicePrimaryCtxRetain(&cPrimaryCtx[i], cDevices[i]) != CUDA_SUCCESS,
				"Failed to retain primary context for device " + std::to_string(i));
		}

		initializedStatus = true;
		initialized = true;
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

	static bool cudaEventFinished(cudaEvent_t &event)
	{
		return CUDAErrorHandler::handleEvent(
			cudaEventQuery(event), "While querying event");
	}

	static void cudaDevicePrefetch(void *pHost, size_t size, int device, cudaStream_t &stream, bool readOnly)
	{
		if (size == 0)
			return;

		// Depending on the access we're prefetching, we will advise the driver to do a shared copy
		cudaMemoryAdvise advice = (readOnly ? cudaMemAdviseSetReadMostly : cudaMemAdviseUnsetReadMostly);
		cudaError_t err = cudaMemAdvise(pHost, size, advice, device);
		CUDAErrorHandler::handle(err, "Advising memory region");

		// Ensure that we have a stream assigned. Stream 0 is special in CUDA and tasks are never launched on it.
		assert(stream != 0);

		// Call a prefetch operation on the same stream that we are going to launch that task on
		err = cudaMemPrefetchAsync(pHost, size, device, stream);
		CUDAErrorHandler::handle(err, "Prefetching memory to device");
	}

	static void memcpy(void *destination, const void *from, size_t count, cudaMemcpyKind kind)
	{
		cudaError_t err = cudaMemcpy(destination, from, count, kind);
		CUDAErrorHandler::handle(err, "Copying memory");
	}
};

#endif // CUDA_FUNCTIONS_HPP
