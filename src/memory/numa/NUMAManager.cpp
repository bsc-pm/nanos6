/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "NUMAManager.hpp"

NUMAManager::directory_t NUMAManager::_directory;
RWSpinLock NUMAManager::_lock;
NUMAManager::alloc_info_t NUMAManager::_allocations;
SpinLock NUMAManager::_allocationsLock;
NUMAManager::bitmask_t NUMAManager::_bitmaskNumaAll;
NUMAManager::bitmask_t NUMAManager::_bitmaskNumaAllActive;
NUMAManager::bitmask_t NUMAManager::_bitmaskNumaAnyActive;
std::atomic<bool> NUMAManager::_trackingEnabled;
ConfigVariable<bool> NUMAManager::_reportEnabled("numa.report");
ConfigVariable<std::string> NUMAManager::_trackingMode("numa.tracking");
