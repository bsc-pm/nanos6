/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/


#include <sstream>
#include <string>

#include <boost/algorithm/string.hpp>

#include <nanos6/config.h>

#include "lowlevel/FatalErrorHandler.hpp"


extern "C" void nanos6_config_assert(const char *config_conditions)
{
	assert(config_conditions != nullptr);

	// NOTE: This implementation only allows asserting the
	// value of version.dependencies. In the future we will
	// include support for the rest of runtime options
#if defined(DISCRETE_DEPS)
	const std::string dependencies("discrete");
#else
	const std::string dependencies("regions");
#endif

	std::istringstream ss(config_conditions);
	std::string currentCondition;

	while (std::getline(ss, currentCondition, ',')) {
		if (currentCondition.empty()) {
			// Silently skip empty conditions
			continue;
		}

		bool equalsOp = true;
		size_t separatorIndex = currentCondition.find("==");
		if (separatorIndex == std::string::npos) {
			separatorIndex = currentCondition.find("!=");
			if (separatorIndex == std::string::npos) {
				FatalErrorHandler::fail("Invalid config assert: compare operator must be == or !=");
			}
			equalsOp = false;
		}

		std::string optionName = currentCondition.substr(0, separatorIndex);
		std::string optionContent = currentCondition.substr(separatorIndex + 2);

		if (optionName.empty() || optionContent.empty()) {
			FatalErrorHandler::fail("Invalid config assert: wrong format in ", currentCondition);
		}

		boost::trim(optionName);
		boost::trim(optionContent);
		boost::to_lower(optionName);
		boost::to_lower(optionContent);

		if (optionName != "version.dependencies") {
			FatalErrorHandler::fail("Config assert only supports checking version.dependencies");
		}

		bool correct;
		if (equalsOp) {
			correct = (dependencies == optionContent);
		} else {
			correct = (dependencies != optionContent);
		}

		if (!correct) {
			FatalErrorHandler::fail("Config assert failed: ", currentCondition);
		}
	}
}
