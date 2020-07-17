/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include "ManagerNUMA.hpp"

ManagerNUMA::directory_t ManagerNUMA::_directory;
RWSpinLock ManagerNUMA::_lock;
