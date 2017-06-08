#ifndef DEPENDENCY_SYSTEM_HPP
#define DEPENDENCY_SYSTEM_HPP

#include "system/RuntimeInfo.hpp"


class DependencySystem {
public:
	static void initialize()
	{
		RuntimeInfo::addEntry("dependency_implementation", "Dependency Implementation", "linear-regions-alternative");
	}
};

#endif // DEPENDENCY_SYSTEM_HPP

