/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023-2024 Barcelona Supercomputing Center (BSC)
*/

#ifndef DIRECTORY_AGENT_HPP
#define DIRECTORY_AGENT_HPP

#include <nanos6/directory.h>

struct DirectoryPage;
class Task;

class DirectoryAgent
{
protected:
    //! Global Agent ID (amongst all agents)
    int _globalId;
    //! Local Agent ID (amongst agents of this device type)
    int _localId;
    oss_device_t _type;

public:
    DirectoryAgent(oss_device_t type, int localId) :
        _localId(localId),
        _type(type)
    {
    }

    inline int getGlobalId() const
    {
        return _globalId;
    }

    inline int getLocalId() const
    {
        return _localId;
    }

    inline void setGlobalId(int globalId)
    {
        _globalId = globalId;
    }

    inline bool isHost() const
    {
        return _type == oss_device_host;
    }

    inline oss_device_t getType() const
    {
        return _type;
    }

    //! Should return true if this agent can copy to `other` directly
    virtual bool canCopyTo(DirectoryAgent *other) = 0;
    //! Should return true if this agent can copy from `other` directly
    virtual bool canCopyFrom(DirectoryAgent *other) = 0;
    //! Initiate a memory copy of page to dst
    virtual void memcpy(DirectoryPage *page, DirectoryAgent *dst, size_t size, void *srcAddress, void *dstAddress) = 0;
    //! Initiate a memory copy of page from src
    virtual void memcpyFrom(DirectoryPage *page, DirectoryAgent *src, size_t size, void *srcAddress, void *dstAddress) = 0;

    //! Should return true if this agent can implement implicit synchronization with a task executed in its device
    //! For example, for CUDA GPUs, we can enqueue copies and then enqueue a task in the same stream, which will guarantee
    //! synchronization without needing to re-schedule the task.
    //! For other agents such as the host, this is not possible, and thus should return false.
    virtual bool canSynchronizeImplicitly()
    {
        return false;
    }

    //! Perform implicitly synchronized memory copy (to the specified task)
    virtual void memcpyFromImplicit(DirectoryPage *, DirectoryAgent *, size_t, void *, void *, Task *)
    {
        return;
    }

    //! Implicitly synchronize the specified task to the currently ongoing memory copy for a specific page.
    virtual void synchronizeOngoing(DirectoryPage *, Task *)
    {
        return;
    }

    virtual void *allocateMemory(size_t size) = 0;
    virtual void freeMemory(void *addr, size_t size) = 0;

    //! Expresses if a Directory Device is capable to implicitly synchronize to a copy
    //! with itself as destination
    virtual bool canSynchronizeOngoingCopies()
    {
        return false;
    }

    virtual ~DirectoryAgent() {}
};

#endif // DIRECTORY_AGENT_HPP
