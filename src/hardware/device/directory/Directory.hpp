/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023-2024 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEVICE_DIRECTORY_HPP
#define DEVICE_DIRECTORY_HPP

#include <atomic>

#include "DirectoryAgent.hpp"
#include "DirectoryEntry.hpp"
#include "HostDirectoryAgent.hpp"
#include "lowlevel/RWSpinLock.hpp"
#include "support/Containers.hpp"

#include <nanos6/directory.h>
#include <nanos6/task-instantiation.h>

class Directory {
private:
	static Directory *_instance;
	static DirectoryAgent *_hostAgent;
	typedef uintptr_t addr_t;
	static std::atomic<bool> _enabled;

	// WARNING! This container is reversed so we can use the lower_bound function
	// Beware when iterating if expecting lower to higher
	typedef Container::map<addr_t, DirectoryEntry, std::greater<addr_t>> directory_map_t;
	directory_map_t _directory;
	RWSpinLock _lock;
	Container::vector<DirectoryAgent *> _agents;

	inline DirectoryEntry *getEntry(addr_t addr)
	{
		_lock.readLock();
		directory_map_t::iterator entry = _directory.lower_bound(addr);

		DirectoryEntry *res = nullptr;

		if (entry != _directory.end())
			res = &(entry->second);

		_lock.readUnlock();
		return res;
	}

	inline int addAgent(DirectoryAgent *agent)
	{
		int id = _agents.size();
		_agents.push_back(agent);
		return id;
	}

	void performFullFlush(Task *taskToUnlock);
	void performPartialFlush(void *location, size_t length, Task *taskToUnlock);
	void flushEntry(DirectoryEntry *entry, Task *taskToUnlock);
	void partiallyFlushEntry(DirectoryEntry *entry, Task *taskToUnlock, void *location, size_t length);

	void readAccess(DirectoryAgent *agent, void *location, size_t length, Task *task, void *&translation);
	void readWriteAccess(DirectoryAgent *agent, void *location, size_t length, Task *task, void *&translation);
	void registerEntry(DirectoryAgent *agent, void *buffer, void *virtualBuffer, size_t size, size_t pageSize);
	void destroyEntry(void *buffer);
	bool processTaskAccesses(DirectoryAgent *agent, Task *task, nanos6_address_translation_entry_t *translationTable, int symbols);

public:
	Directory() :
		_directory(),
		_lock(),
		_agents()
	{
	}

	static void initialize()
	{
		_instance = MemoryAllocator::newObject<Directory>();
		_hostAgent = MemoryAllocator::newObject<HostDirectoryAgent>();

		registerDevice(_hostAgent);
		assert(_hostAgent->getGlobalId() == 0);
	}

	static void shutdown()
	{
		delete _instance;
	}

	static void registerDevice(DirectoryAgent *agent)
	{
		agent->setGlobalId(_instance->addAgent(agent));
	}

	static bool preTaskExecution(
		DirectoryAgent *agent,
		Task *task,
		nanos6_address_translation_entry_t *translationTable,
		int symbols);

	static DirectoryAgent *getHostAgent()
	{
		return _hostAgent;
	}

	static bool isEnabled()
	{
		return _enabled.load(std::memory_order_relaxed);
	}

	static void *deviceAlloc(oss_device_t device, int index, size_t size, size_t stride);
	static void deviceFree(void *address);

	static bool flush(Task *task);
	static bool flushTaskDependencies(Task *task);
};

#endif // DEVICE_DIRECTORY_HPP