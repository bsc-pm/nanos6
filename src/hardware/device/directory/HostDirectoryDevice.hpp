/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef HOST_DIRECTORY_DEVICE_HPP
#define HOST_DIRECTORY_DEVICE_HPP

#include "DirectoryDevice.hpp"

#include <MemoryAllocator.hpp>

#include <nanos6/directory.h>

// This class is special since there is only one host
// It cannot copy to/from anywhere
class HostDirectoryDevice : public DirectoryDevice
{
public:
    bool canCopyTo(DirectoryDevice *) override
    {
        return false;
    }

    bool canCopyFrom(DirectoryDevice *) override
    {
        return false;
    }

    void memcpy(DirectoryPage *, DirectoryDevice *, size_t, void *, void *) override
    {
    }

    void memcpyFrom(DirectoryPage *, DirectoryDevice *, size_t, void *, void *) override
    {
    }

    void *allocateMemory(size_t size) override {
        return MemoryAllocator::alloc(size);
    }

    void freeMemory(void *addr, size_t size) override {
        MemoryAllocator::free(addr, size);
    }
};

#endif // HOST_DIRECTORY_DEVICE_HPP
