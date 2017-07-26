/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DEPENDENCY_SYSTEM_HPP
#define DEPENDENCY_SYSTEM_HPP

#include "system/RuntimeInfo.hpp"


class DependencySystem {
public:
	static void initialize()
	{
		RuntimeInfo::addEntry("dependency_implementation", "Dependency Implementation", "linear-regions-fragmented");
	}
};

#endif // DEPENDENCY_SYSTEM_HPP

