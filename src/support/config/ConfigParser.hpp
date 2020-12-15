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

#include "lowlevel/EnvironmentVariable.hpp"
#include "lowlevel/FatalErrorHandler.hpp"
#include "support/StringSupport.hpp"


class ConfigParser {
private:
	//! Map to store option name and string value
	typedef std::unordered_map<std::string, std::string> environment_config_map_t;

	//! Configuration file data
	toml::value _data;

	//! Configuration override envar string values
	environment_config_map_t _environmentConfig;

private:
	//! \brief Get element by key in config file
	//!
	//! \param key The key of the element
	//!
	//! \returns The element
	inline toml::value findKey(const std::string &key)
	{
		std::string tmp;
		std::istringstream ss(key);
		toml::value *it = &_data;

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

	//! \brief Parse the config envar
	//!
	//! This function reads the NANOS6_CONFIG_OVERRIDE envar
	//! and stores the options and values into the map of
	//! envar options
	inline void parseEnvironmentConfig()
	{
		const EnvironmentVariable<std::string> configOverride("NANOS6_CONFIG_OVERRIDE", "");

		if (!configOverride.getValue().empty()) {
			processOptions(configOverride.getValue(), /* are assignments */ false,
				[&](const std::string &option, const std::string &value, const std::string &) {
					_environmentConfig[option] = value;
				}
			);
		}
	}

	//! \brief Private constructor of config parser
	//!
	//! This function parses both the config file and the
	//! config override envar
	ConfigParser() :
		_environmentConfig()
	{
		// Get the config file path computed by the loader
		const char *_nanos6_config_path = (const char *) dlsym(nullptr, "_nanos6_config_path");
		assert(_nanos6_config_path != nullptr);

		try {
			// Parse the config file
			_data = toml::parse(_nanos6_config_path);
		} catch (std::runtime_error &error) {
			FatalErrorHandler::fail("Error while opening the configuration file found in ",
				std::string(_nanos6_config_path), ". Inner error: ", error.what());
		} catch (toml::syntax_error &error) {
			FatalErrorHandler::fail("Configuration syntax error: ", error.what());
		}

		// Parse the config envar
		parseEnvironmentConfig();
	}

public:
	//! \brief Try to get the config value of an option
	//!
	//! This function first checks the config override envar and then
	//! the config file. This function may fail if the conversion of
	//! value to the template type is not possible
	//!
	//! \param option The option name
	//! \param value The value of the option if found
	//!
	//! \returns Whether the option was found
	template <typename T>
	inline bool get(const std::string &option, T &value)
	{
		// First we will try to find the corresponding override option
		auto optionIt = _environmentConfig.find(option);
		if (optionIt != _environmentConfig.end()) {
			if (StringSupport::parse<T>(optionIt->second, value, std::boolalpha)) {
				// Correctly parsed
				return true;
			} else {
				FatalErrorHandler::fail("Configuration override for option ",
					option, " found but value '", optionIt->second,
					"' could not be cast to ", typeid(T).name());
			}
		}

		// Check the config file
		toml::value element = findKey(option);

		if (element.is_uninitialized()) {
			// The option was not found
			return false;
		}

		try {
			// Get and convert value
			value = toml::get<T>(element);
		} catch (toml::type_error &error) {
			FatalErrorHandler::fail("Expecting type ", typeid(T).name(),
				" in configuration option ", option, ", but found ",
				toml::stringize(element.type()), " instead.");
		}

		return true;
	}

	//! \brief Try to get the config values of a list option
	//!
	//! This function only checks the config file since lists are not
	//! supported by the override envar. This function may fail if the
	//! conversion of values to the template type is not possible
	//!
	//! \param option The option name
	//! \param values The values' vector of the option if found
	//!
	//! \returns Whether the option was found
	template <typename T>
	inline bool getList(const std::string &option, std::vector<T> &values)
	{
		// Check the config file
		toml::value element = findKey(option);

		if (element.is_uninitialized()) {
			// The option was not found
			return false;
		}

		try {
			// Get and convert value
			values = toml::get<std::vector<T>>(element);
		} catch (toml::type_error &error) {
			FatalErrorHandler::fail("Expecting type list(", typeid(T).name(),
				") in configuration option ", option, ", but found ",
				toml::stringize(element.type()), " instead.");
		}

		return true;
	}

	//! \brief Process conditions/assignments of options in string format
	//!
	//! This function parses a comma-separated string containing conditions or
	//! assignments of options and calls a processor function for each one
	//!
	//! \param options A comma-separated string containing the conditions/assignments
	//! \param areConditions Whether string contains conditions
	//! \param processor A function that takes three strings: the option name, the
	//! option value and the option operator
	template <typename ProcessorType>
	static inline void processOptions(const std::string &options, bool areConditions, ProcessorType processor)
	{
		std::istringstream ss(options);
		std::string currentOption;

		// Traverse the conditions or assignments
		while (std::getline(ss, currentOption, ',')) {
			if (currentOption.empty()) {
				// Silently skip empty options
				continue;
			}

			// Find the operator
			std::string op;
			size_t separatorIndex = StringSupport::findOperator(currentOption, areConditions, op);
			if (separatorIndex == std::string::npos) {
				FatalErrorHandler::fail("Invalid config option format");
			}

			// Retrieve the option name and value
			std::string optionName = currentOption.substr(0, separatorIndex);
			std::string optionValue = currentOption.substr(separatorIndex + op.length());

			if (optionName.empty()) {
				FatalErrorHandler::fail("Invalid config option: option name cannot be empty");
			}

			if (optionValue.empty()) {
				FatalErrorHandler::fail("Invalid config option: value cannot be empty in option ", optionName);
			}

			// Trim both option and value
			boost::trim(optionName);
			boost::trim(optionValue);

			// All config options are in lowercase
			boost::to_lower(optionName);

			// Process the condition or assignment
			processor(optionName, optionValue, op);
		}
	}

	//! \brief Get the config parser
	//!
	//! \returns The config parser
	static inline ConfigParser &getParser()
	{
		static ConfigParser configParser;
		return configParser;
	}
};

#endif // CONFIG_PARSER_HPP
