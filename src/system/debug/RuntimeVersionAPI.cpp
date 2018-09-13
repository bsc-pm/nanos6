/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6/debug.h>
#include "version/VersionInfo.hpp"


char const *nanos6_get_runtime_version(void)
{
	return nanos6_version;
}

char const *nanos6_get_runtime_copyright(void)
{
	return nanos6_copyright;
}

char const *nanos6_get_runtime_license(void)
{
	return nanos6_license;
}

char const *nanos6_get_runtime_full_license(void)
{
	return nanos6_full_license;
}

char const *nanos6_get_runtime_branch(void)
{
	return nanos6_branch;
}

char const *nanos6_get_runtime_patches(void)
{
	return nanos6_patches;
}

char const *nanos6_get_runtime_compiler_version(void)
{
	return nanos6_compiler_version;
}

char const *nanos6_get_runtime_compiler_flags(void)
{
	return nanos6_compiler_flags;
}


