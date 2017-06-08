#include "RuntimeInfoEssentials.hpp"
#include "system/RuntimeInfo.hpp"

#include "version/VersionInfo.hpp"


void RuntimeInfoEssentials::initialize()
{
	RuntimeInfo::addEntry("version", "Runtime Version", NANOS_VERSION);
	RuntimeInfo::addEntry("branch", "Runtime Branch", NANOS_BRANCH);
	RuntimeInfo::addEntry("compiler_version", "Runtime Compiler Version", CXX_VERSION);
	RuntimeInfo::addEntry("compiler_flags", "Runtime Compiler Flags", CXXFLAGS);
}

