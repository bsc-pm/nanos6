/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include "CPUManager.hpp"


CPUManagerInterface *CPUManager::_cpuManager;
ConfigVariable<bool> CPUManager::_dlbEnabled("dlb.enabled", false);
