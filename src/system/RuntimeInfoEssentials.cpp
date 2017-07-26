/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include "RuntimeInfoEssentials.hpp"
#include "system/RuntimeInfo.hpp"

#include "version/VersionInfo.hpp"


void RuntimeInfoEssentials::initialize()
{
	RuntimeInfo::addEntry("version", "Runtime Version", nanos6_version);
	RuntimeInfo::addEntry("branch", "Runtime Branch", nanos6_branch);
	RuntimeInfo::addEntry("compiler_version", "Runtime Compiler Version", nanos6_compiler_version);
	RuntimeInfo::addEntry("compiler_flags", "Runtime Compiler Flags", nanos6_compiler_flags);
}

