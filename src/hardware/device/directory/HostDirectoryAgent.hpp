/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023-2024 Barcelona Supercomputing Center (BSC)
*/

#ifndef HOST_DIRECTORY_AGENT_HPP
#define HOST_DIRECTORY_AGENT_HPP

#include "DirectoryAgent.hpp"

#include <MemoryAllocator.hpp>

#include <nanos6/directory.h>

// This class is special since there is only one host
// It cannot copy to/from anywhere
class HostDirectoryAgent : public DirectoryAgent
{
public:
    HostDirectoryAgent() : DirectoryAgent(oss_device_host, 0)
    {
    }

    bool canCopyTo(DirectoryAgent *) override
    {
        return false;
    }

    bool canCopyFrom(DirectoryAgent *) override
    {
        return false;
    }

    void memcpy(DirectoryPage *, DirectoryAgent *, size_t, void *, void *) override
    {
    }

    void memcpyFrom(DirectoryPage *, DirectoryAgent *, size_t, void *, void *) override
    {
    }

    void *allocateMemory(size_t size) override {
        return MemoryAllocator::alloc(size);
    }

    void freeMemory(void *addr, size_t size) override {
        MemoryAllocator::free(addr, size);
    }
};

#endif // HOST_DIRECTORY_AGENT_HPP
