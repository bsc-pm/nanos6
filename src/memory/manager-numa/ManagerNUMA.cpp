/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "ManagerNUMA.hpp"

ManagerNUMA::directory_t ManagerNUMA::_directory;
RWSpinLock ManagerNUMA::_lock;
ManagerNUMA::alloc_info_t ManagerNUMA::_allocations;
SpinLock ManagerNUMA::_allocationsLock;
size_t ManagerNUMA::_numNumaAll;
size_t ManagerNUMA::_numNumaAllActive;
size_t ManagerNUMA::_numNumaAnyActive;
size_t ManagerNUMA::_maxCpusPerNuma;
ManagerNUMA::bitmask_t ManagerNUMA::_bitmaskNumaAll;
ManagerNUMA::bitmask_t ManagerNUMA::_bitmaskNumaAllActive;
ManagerNUMA::bitmask_t ManagerNUMA::_bitmaskNumaAnyActive;
std::vector<size_t> ManagerNUMA::_cpusPerNumaNode;
bool ManagerNUMA::_reportEnabled;
#ifndef NDEBUG
std::atomic<size_t> ManagerNUMA::_totalBytes;
std::atomic<size_t> ManagerNUMA::_totalQueries;
#endif
