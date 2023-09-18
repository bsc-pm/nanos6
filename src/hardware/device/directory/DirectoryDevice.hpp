/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef DIRECTORY_DEVICE_HPP
#define DIRECTORY_DEVICE_HPP

#include <nanos6/directory.h>

struct DirectoryPage;

class DirectoryDevice
{
protected:
    int _id;
    oss_device_t _type;

public:
    inline int getId() const
    {
        return _id;
    }

    inline void setId(int id)
    {
        _id = id;
    }

    inline bool isHost() const
    {
        return _id == 0;
    }

    inline oss_device_t getType() const
    {
        return _type;
    }

    virtual bool canCopyTo(DirectoryDevice *other) = 0;
    virtual bool canCopyFrom(DirectoryDevice *other) = 0;
    virtual void memcpy(DirectoryPage *page, DirectoryDevice *dst, size_t size, void *srcAddress, void *dstAddress) = 0;
    virtual void memcpyFrom(DirectoryPage *page, DirectoryDevice *src, size_t size, void *srcAddress, void *dstAddress) = 0;
    virtual void *allocateMemory(size_t size) = 0;

    virtual ~DirectoryDevice() {}
};

#endif // DIRECTORY_DEVICE_HPP
