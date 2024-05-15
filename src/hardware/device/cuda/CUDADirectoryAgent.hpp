/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023-2024 Barcelona Supercomputing Center (BSC)
*/

#ifndef CUDA_DIRECTORY_AGENT_HPP
#define CUDA_DIRECTORY_AGENT_HPP

#include "CUDAFunctions.hpp"
#include "hardware/device/directory/DirectoryAgent.hpp"
#include "lowlevel/PaddedTicketSpinLock.hpp"
#include "support/Containers.hpp"

#include <cassert>

class CUDAAccelerator;
class Task;

class CUDADirectoryAgent : public DirectoryAgent {
private:
	CUDAAccelerator *_accelerator;
	cudaStream_t _directoryStream;

	struct OngoingCopy {
		cudaEvent_t event;
		DirectoryPage *page;
		int destinationDevice;
	};
	Container::deque<OngoingCopy> _pendingEvents;
	Container::deque<OngoingCopy> _pendingEventsCopy;

	typedef PaddedTicketSpinLock<int> spinlock_t;
	spinlock_t _lock;

	void memcpyFromImpl(DirectoryPage *page, DirectoryAgent *src, size_t size, void *srcAddress, void *dstAddress, cudaStream_t stream);

public:
	CUDADirectoryAgent(CUDAAccelerator *accelerator);

	bool canCopyTo(DirectoryAgent *other) override
	{
		return other->getType() == oss_device_cuda || other->getType() == oss_device_host;
	}

	bool canCopyFrom(DirectoryAgent *other) override
	{
		return other->getType() == oss_device_cuda || other->getType() == oss_device_host;
	}

	void memcpy(DirectoryPage *page, DirectoryAgent *dst, size_t size, void *srcAddress, void *dstAddress) override;

	void memcpyFrom(DirectoryPage *page, DirectoryAgent *src, size_t size, void *srcAddress, void *dstAddress) override;

	void *allocateMemory(size_t size) override
	{
		return CUDAFunctions::malloc(size);
	}

	void freeMemory(void *addr, size_t) override
	{
		CUDAFunctions::free(addr);
	}

	bool canSynchronizeImplicitly() override
	{
		return true;
	}

	void memcpyFromImplicit(DirectoryPage *page, DirectoryAgent *src, size_t size, void *srcAddress, void *dstAddress, Task *task) override;

	void synchronizeOngoing(DirectoryPage *page, Task *task) override;

	bool canSynchronizeOngoingCopies() override
	{
		return true;
	}

	void processEvents();
};

#endif // CUDA_DIRECTORY_AGENT_HPP
