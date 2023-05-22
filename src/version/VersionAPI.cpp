/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2023 Barcelona Supercomputing Center (BSC)
*/

#include <nanos6.h>

#include <cassert>
#include <cstdint>
#include <string>
#include <sstream>
#include <vector>

#include "lowlevel/FatalErrorHandler.hpp"

// The current API supported is family 0 with version 1.0
static constexpr uint64_t GENERAL_API_FAMILY = 0;
static constexpr uint64_t GENERAL_API_MAJOR = 1;
static constexpr uint64_t GENERAL_API_MINOR = 0;

extern "C" void nanos6_check_version(
	uint64_t size, nanos6_version_t *versions, const char *source
) {
	assert(source != nullptr);

	std::vector<std::string> errors;

	for (uint64_t v = 0; v < size; ++v) {
		std::ostringstream oss;

		// The version family is not even recognized
		if (versions[v].family != GENERAL_API_FAMILY)
			oss << "Family " << versions[v].family << " not recognized";
		// The version family is recognized but not compatible
		else if (versions[v].major_version != GENERAL_API_MAJOR || versions[v].minor_version > GENERAL_API_MINOR)
			oss << "Family " << versions[v].family << " requires " << versions[v].major_version << "."
				<< versions[v].minor_version << ", but runtime supports " << GENERAL_API_MAJOR << "."
				<< GENERAL_API_MINOR;
		// The version is supported
		else
			continue;

		errors.push_back(oss.str());
	}

	// If any incompatibilities were found, report them and abort the execution
	if (!errors.empty()) {
		std::ostringstream oss;

		for (uint64_t e = 0; e < errors.size(); ++e)
			oss << "\n\t" << (e + 1) << ". " << errors[e];

		FatalErrorHandler::fail("Found API version incompatibilities in ", source, ":", oss.str());
	}
}
