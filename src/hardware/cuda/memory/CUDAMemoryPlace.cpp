/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2018 Barcelona Supercomputing Center (BSC)
*/

#include "CUDAMemoryPlace.hpp"
#include "CUDAUnifiedMemory.hpp"

#include "lowlevel/FatalErrorHandler.hpp"

CUDAMemoryPlace *CUDAMemoryPlace::createCUDAMemory(std::string const &memoryMode, int index, cudaDeviceProp &properties)
{
	FatalErrorHandler::failIf(memoryMode != "unified" && memoryMode != "default", "Only \"unified\" memory mode is supported");
	return new CUDAUnifiedMemory(index, properties);
}

