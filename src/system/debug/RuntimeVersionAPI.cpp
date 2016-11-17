#include <nanos6/debug.h>
#include "version/VersionInfo.hpp"


char const *nanos_get_runtime_version(void)
{
	return NANOS_VERSION;
}

char const *nanos_get_runtime_branch(void)
{
	return NANOS_BRANCH;
}

char const *nanos_get_runtime_compiler_version(void)
{
	return CXX_VERSION;
}

char const *nanos_get_runtime_compiler_flags(void)
{
	return CXXFLAGS;
}


