/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CONFIG_PARSER_HPP
#define CONFIG_PARSER_HPP

#include <cassert>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include "toml.hpp"
#pragma GCC diagnostic pop

#include "lowlevel/FatalErrorHandler.hpp"
#include "lowlevel/EnvironmentVariable.hpp"

class ConfigParser {
	toml::value data;

	typedef std::unordered_map<std::string, std::string> environment_config_map_t;
	environment_config_map_t _environmentConfig;

	toml::value findKey(std::string key)
	{
		std::string tmp;
		std::istringstream ss(key);
		toml::value *it = &data;

		while (std::getline(ss, tmp, '.')) {
			toml::value &current = *it;
			if (!current.is_table())
				return toml::value();

			if (!current.contains(tmp))
				return toml::value();

			toml::value &next = toml::find(current, tmp);
			it = &next;
		}

		return *it;
	}

	void parseEnvironmentConfig()
	{
		const EnvironmentVariable<std::string> configOverride("NANOS6_CONFIG_OVERRIDE", "");

		if (!configOverride.getValue().empty()) {
			std::istringstream ss(configOverride);
			std::string currentDirective;

			while (std::getline(ss, currentDirective, ',')) {
				if (currentDirective.empty()) {
					// Silently skip empty directives
					continue;
				}

				size_t separatorIndex = currentDirective.find('=');
				if (separatorIndex == std::string::npos) {
					FatalErrorHandler::warn("Invalid config option: directive must follow format 'option=value'");
					continue;
				}

				std::string directiveName = currentDirective.substr(0, separatorIndex);
				std::string directiveContent = currentDirective.substr(separatorIndex + 1);

				if (directiveName.empty()) {
					FatalErrorHandler::warn("Invalid config option: directive name cannot be empty");
					continue;
				}

				if (directiveContent.empty()) {
					FatalErrorHandler::warn("Invalid config option: directive content cannot be empty in option ", directiveName);
					continue;
				}

				if (_environmentConfig.find(directiveName) != _environmentConfig.end()) {
					FatalErrorHandler::warn("Ignoring repeated config option ", directiveName);
					continue;
				}

				// All config options are in lowercase
				boost::trim(directiveName);
				boost::to_lower(directiveName);

				_environmentConfig.emplace(directiveName, directiveContent);
			}
		}
	}

	template <typename T>
	bool parseString(std::string str, T &result)
	{
		std::istringstream iss(str);
		T tmp = result;
		iss >> std::boolalpha >> tmp;

		if (!iss.fail()) {
			result = tmp;
			return true;
		}

		return false;
	}

public:
	ConfigParser() :
		_environmentConfig()
	{
		const char *_nanos6_config_path = (const char *)dlsym(nullptr, "_nanos6_config_path");
		assert(_nanos6_config_path != nullptr);

		try {
			data = toml::parse(_nanos6_config_path);
		} catch (std::runtime_error &error) {
			FatalErrorHandler::fail("Error while opening the configuration file found in ",
				std::string(_nanos6_config_path),
				". Inner error: ",
				error.what());
		} catch (toml::syntax_error &error) {
			FatalErrorHandler::fail("Configuration syntax error: ", error.what());
		}

		parseEnvironmentConfig();
	}

	template <typename T>
	void get(std::string key, T &value, bool &found)
	{
		// First we will try to find the corresponding override
		environment_config_map_t::iterator option = _environmentConfig.find(key);
		if (option != _environmentConfig.end()) {
			if (parseString<T>(option->second, value)) {
				found = true;
				return;
			} else {
				FatalErrorHandler::warn(
					"Configuration override for ",
					key,
					" found but value '",
					option->second,
					"' could not be cast to ",
					typeid(T).name(),
					", falling back to config file");
			}
		}

		toml::value element = findKey(key);

		if (element.is_uninitialized()) {
			found = false;
			return;
		}

		try {
			value = toml::get<T>(element);
		} catch (toml::type_error &error) {
			found = false;

			FatalErrorHandler::warn(
				"Expecting type ", typeid(T).name(), " in configuration key ",
				key,
				", but found ",
				toml::stringize(element.type()),
				" instead.");
		}

		found = true;
	}

	template <typename T>
	void getList(std::string key, std::vector<T> &value, bool &found)
	{
		toml::value element = findKey(key);

		if (element.is_uninitialized()) {
			found = false;
			return;
		}

		try {
			value = toml::get<std::vector<T>>(element);
		} catch (toml::type_error &error) {
			found = false;

			FatalErrorHandler::warn(
				"Expecting type list(", typeid(T).name(), ") in configuration key ",
				key,
				", but found ",
				toml::stringize(element.type()),
				" instead.");
		}

		found = true;
	}
};

#endif // CONFIG_PARSER_HPP
