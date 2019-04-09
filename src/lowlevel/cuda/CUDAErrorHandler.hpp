/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_FATAL_ERROR_HANDLER_HPP
#define CUDA_FATAL_ERROR_HANDLER_HPP

#include <cuda_runtime_api.h>

#include "lowlevel/FatalErrorHandler.hpp"

class CUDAErrorHandler: public FatalErrorHandler {
private:
	
	static inline void printCUDAError(cudaError_t err, std::ostringstream &oss)
	{
		std::string errName = cudaGetErrorName(err);
		std::string errReason = cudaGetErrorString(err);
		
		switch (err) {
			case cudaErrorIllegalAddress:
				oss << errName << ": Illegal Address accessed by CUDA, managed memory allocation is needed when using unified memory mode"; 
				break;
			default:
				oss << errName << ": " << errReason;
				break;
		}
	}
	
public:
	
	template<typename... TS>
	static inline void handle(cudaError_t err, TS... reasonParts)
	{
		if (__builtin_expect(err == cudaSuccess, 1)) {
			return;
		}
		
		std::ostringstream oss;
		
		printCUDAError(err, oss);
		emitReasonParts(oss, reasonParts...);
		oss << std::endl;
		
		{
			std::lock_guard<SpinLock> guard(_lock);
			std::cerr << oss.str();
		}
		
#ifndef NDEBUG
		abort();
#else
		exit(1);
#endif
	}

	template<typename... TS>
	static inline bool handleEvent(cudaError_t err, TS... reasonParts)
	{
		if (__builtin_expect(err == cudaErrorNotReady || err == cudaSuccess, 1)) {
			if (err == cudaErrorNotReady) {
				return false;
			} else {
				return true;
			}
		}
		
		std::ostringstream oss;
		
		printCUDAError(err, oss);
		emitReasonParts(oss, reasonParts...);
		oss << std::endl;
		
		{
			std::lock_guard<SpinLock> guard(_lock);
			std::cerr << oss.str();
		}
		
#ifndef NDEBUG
		abort();
#else
		exit(1);
#endif
	}
	
	template<typename... TS>
	static inline void warnIf(bool failure, TS... reasonParts)
	{
		if (__builtin_expect(!failure, 1)) {
			return;
		}
		
		std::ostringstream oss;
		oss << "Warning: ";
		emitReasonParts(oss, reasonParts...);
		oss << std::endl;
		
		{
			std::lock_guard<SpinLock> guard(_lock);
			std::cerr << oss.str();
		}
	}
};

#endif //CUDA_FATAL_ERROR_HANDLER
