/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef DIRECTORY_HPP
#define DIRECTORY_HPP

#include "DirectoryEntry.hpp"
#include "lowlevel/RWSpinLock.hpp"
#include "support/Containers.hpp"

class Directory {
private:
	static Directory *_instance;
	typedef uintptr_t addr_t;

	Container::map<addr_t, DirectoryEntry> _directory;
	RWSpinLock _lock;

public:
	static void initialize() {
		_instance = MemoryAllocator::newObject<Directory>();
	}

	static void shutdown() {
		delete _instance;
	}
};

#endif // DIRECTORY_HPP