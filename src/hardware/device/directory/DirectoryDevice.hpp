/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef DIRECTORY_DEVICE_HPP
#define DIRECTORY_DEVICE_HPP

#include <nanos6/directory.h>

struct DirectoryPage;
class Task;

class DirectoryDevice
{
protected:
    int _id;
    oss_device_t _type;

public:
    DirectoryDevice(oss_device_t type) :
        _type(type)
    {
    }

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

    virtual bool canSynchronizeImplicitely()
    {
        return false;
    }

    virtual void memcpyFromImplicit(DirectoryPage *, DirectoryDevice *, size_t, void *, void *, Task *)
    {
        return;
    }

    virtual void synchronizeOngoing(DirectoryPage *, Task *)
    {
        return;
    }

    virtual void *allocateMemory(size_t size) = 0;
    virtual void freeMemory(void *addr, size_t size) = 0;

    // Expresses if a Directory Device is capable to synchronize to a copy
    // with itself as destination
    virtual bool canSynchronizeOngoingCopies()
    {
        return false;
    }

    virtual ~DirectoryDevice() {}
};

#endif // DIRECTORY_DEVICE_HPP
