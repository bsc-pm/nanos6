/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include "CPUManager.hpp"


CPUManagerInterface *CPUManager::_cpuManager;
EnvironmentVariable<bool> CPUManager::_dlbEnabled("NANOS6_ENABLE_DLB", false);
