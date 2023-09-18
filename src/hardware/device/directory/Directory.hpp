/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef DIRECTORY_HPP
#define DIRECTORY_HPP

#include "DirectoryDevice.hpp"
#include "DirectoryEntry.hpp"
#include "HostDirectoryDevice.hpp"
#include "lowlevel/RWSpinLock.hpp"
#include "support/Containers.hpp"

#include <nanos6/directory.h>
#include <nanos6/task-instantiation.h>

class Directory {
private:
	static Directory *_instance;
	static DirectoryDevice *_hostDevice;
	typedef uintptr_t addr_t;

	Container::map<addr_t, DirectoryEntry> _directory;
	RWSpinLock _lock;
	Container::vector<DirectoryDevice *> _devices;

	inline DirectoryEntry *getEntry(addr_t addr)
	{
		_lock.readLock();
		Container::map<addr_t, DirectoryEntry>::iterator entry = _directory.lower_bound(addr);

		DirectoryEntry *res = nullptr;

		if (entry != _directory.end())
			res = &(entry->second);

		_lock.readUnlock();
		return res;
	}

	inline int addDevice(DirectoryDevice *device)
	{
		int id = _devices.size();
		_devices.push_back(device);
		return id;
	}

	void readAccess(DirectoryDevice *device, void *location, size_t length, Task *task, void *&translation);
	void readWriteAccess(DirectoryDevice *device, void *location, size_t length, Task *task, void *&translation);
	void registerEntry(DirectoryDevice *device, void *buffer, size_t size, size_t pageSize);
	bool processTaskAccesses(DirectoryDevice *device, Task *task, nanos6_address_translation_entry_t *translationTable, int symbols);

public:
	Directory() :
		_directory(),
		_lock(),
		_devices()
	{
	}

	static void initialize()
	{
		_instance = MemoryAllocator::newObject<Directory>();
		_hostDevice = MemoryAllocator::newObject<HostDirectoryDevice>();

		registerDevice(_hostDevice);
		assert(_hostDevice->getId() == 0);
	}

	static void shutdown()
	{
		delete _instance;
	}

	static void registerDevice(DirectoryDevice *device)
	{
		device->setId(_instance->addDevice(device));
	}

	static bool preTaskExecution(
		DirectoryDevice *device,
		Task *task,
		nanos6_address_translation_entry_t *translationTable,
		int symbols);

	static DirectoryDevice *getHostDevice()
	{
		return _hostDevice;
	}

	static void *deviceAlloc(oss_device_t device, int index, size_t size, size_t stride);
};

#endif // DIRECTORY_HPP