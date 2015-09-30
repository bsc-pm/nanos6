#include "api/nanos6_debug_interface.h"


#include "version/VersionInfo.hpp"


char const *nanos_get_runtime_version()
{
	return NANOS_VERSION;
}

char const *nanos_get_runtime_branch()
{
	return NANOS_BRANCH;
}

char const *nanos_get_runtime_compiler_version()
{
	return CXX_VERSION;
}

char const *nanos_get_runtime_compiler_flags()
{
	return CXXFLAGS;
}


