/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/


#include <string>

#include <nanos6/config.h>

#include "support/config/ConfigChecker.hpp"


extern "C" void nanos6_config_assert(const char *config_conditions)
{
	assert(config_conditions != nullptr);

	std::string conditions(config_conditions);
	if (conditions.empty())
		return;

	ConfigChecker::addAssertConditions(conditions);
}
