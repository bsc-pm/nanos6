/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6.h>

#include "lowlevel/FatalErrorHandler.hpp"

void *nanos6_get_reduction_storage1(__attribute__((unused)) void *original,
	__attribute__((unused)) long dim1size,
	__attribute__((unused)) long dim1start,
	__attribute__((unused)) long dim1end)
{
	FatalErrorHandler::fail("This version requires symbol translation, calling nanos6_get_reduction_storage1 is incorrect");
	return nullptr;
}
