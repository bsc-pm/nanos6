/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#include <iostream>
#include <sstream>

#include "ConfigCentral.hpp"
#include "ConfigParser.hpp"


void ConfigParser::checkFileOptions(
	const toml::value &node,
	std::string key,
	std::unordered_set<std::string> &invalidOptions
) {
	if (!node.is_table()) {
		// Check the complete file option
		if (!ConfigCentral::existsOption(key)) {
			invalidOptions.emplace(key);
		}
		return;
	}

	// The first key should not be preceeded by a dot
	if (!key.empty()) {
		key += ".";
	}

	// Recursively check all subkeys
	for (const auto &element : node.as_table()) {
		const std::string &subkey = element.first;
		assert(!subkey.empty());

		checkFileOptions(element.second, key + subkey, invalidOptions);
	}
}

void ConfigParser::checkFileAndOverrideOptions()
{
	std::unordered_set<std::string> invalidOptions;

	// Check file options recursively
	checkFileOptions(_data, "", invalidOptions);

	// Check override options
	for (const auto &option : _environmentConfig) {
		const std::string &name = option.first;
		assert(!name.empty());

		if (!ConfigCentral::existsOption(name)) {
			invalidOptions.emplace(name);
		}
	}

	if (!invalidOptions.empty()) {
		// Get the default config file path computed by the loader
		const char *_nanos6_default_config_path = (const char *) dlsym(nullptr, "_nanos6_default_config_path");
		assert(_nanos6_default_config_path != nullptr);

		std::ostringstream oss;
		oss << "Detected invalid options in the config file or in the NANOS6_CONFIG_OVERRIDE environment variable." << std::endl;
		oss << "The invalid options are:" << std::endl;

		oss << "\t";
		for (const std::string &option : invalidOptions) {
			oss << option << "  ";
		}
		oss << std::endl << std::endl;

		oss << "The config file options may have been updated in this runtime version.";
		oss << " Please check the differences with the updated default config file:" << std::endl;
		oss << "\t" << _nanos6_default_config_path << std::endl;

		FatalErrorHandler::fail(oss.str());
	}
}
