/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2021 Barcelona Supercomputing Center (BSC)
*/

#ifndef CONFIG_CENTRAL_HPP
#define CONFIG_CENTRAL_HPP


#include <string>
#include <unordered_map>
#include <vector>

#include <boost/variant.hpp>
#include <boost/variant/get.hpp>

#include "ConfigParser.hpp"
#include "ConfigSupport.hpp"


//! Static class that stores all the valid config options and their
//! defaults. This is the only class the ones that accesses the config
//! parser to retrieve option values from the config envar/file. All
//! new options must be registered in this class. See the function
//! ConfigCentral::initialize for more information
class ConfigCentral {
public:
	//! Option kinds (run-time)
	typedef ConfigOptionType::OptionKind OptionKind;

	//! Allowed types (compile-time) in config central
	typedef ConfigOptionType::bool_t bool_t;
	typedef ConfigOptionType::float_t float_t;
	typedef ConfigOptionType::integer_t integer_t;
	typedef ConfigOptionType::string_t string_t;
	typedef ConfigOptionType::memory_t memory_t;

private:
	//! Descriptor of a valid config option
	struct ConfigDescriptor {
		//! The option kind (run-time)
		OptionKind _kind;

		//! Whether it is a list option
		bool _isList;
	};

	//! Type that encapsulates all allowed config types
	typedef boost::variant<bool_t, float_t, integer_t, string_t, memory_t> variant_t;

	//! Map to store option descriptors
	typedef std::unordered_map<std::string, ConfigDescriptor> descriptors_t;

	//! Map to store option defaults
	typedef std::unordered_map<std::string, variant_t> defaults_t;

	//! Map to store option list defaults
	typedef std::unordered_map<std::string, std::vector<variant_t>> list_defaults_t;

	//! Descriptors of valid options
	descriptors_t _descriptors;

	//! Default values of regular options
	defaults_t _defaults;

	//! Default values of list options
	list_defaults_t _listDefaults;

private:
	//! \brief Private constructor of config central
	ConfigCentral();

	//! \brief Get the config central
	//!
	//! The first call creates the config central object. All config
	//! central class members need to be created dynamically since this
	//! class is accessed by static config variables. Having these class
	//! members as regular static fields in the class would produce the
	//! well-known static order fiasco
	//!
	//! \returns The config central
	static inline ConfigCentral &getCentral()
	{
		static ConfigCentral configCentral;
		return configCentral;
	}

	//! \brief Register regular option of an accepted type
	//!
	//! This function should be called only during initialization and is
	//! not thread safe
	//!
	//! \param option The option name
	//! \param defaultValue The default value of the option
	template <typename T>
	inline void registerOption(const std::string &option, const T &defaultValue)
	{
		assert(_descriptors.find(option) == _descriptors.end());

		// Get the option kind
		OptionKind kind = ConfigOptionType::getOptionKind<T>();

		// Add the option descriptor
		_descriptors[option] = { kind, false };

		// Set the option default
		_defaults[option] = defaultValue;
	}

	//! \brief Register list option of an accepted type
	//!
	//! This function should be called only during initialization and is
	//! not thread safe
	//!
	//! \param option The option name
	//! \param defaultValues The default list of values
	template <typename T>
	inline void registerOption(
		const std::string &option,
		const std::initializer_list<T> &defaultValues
	) {
		assert(_descriptors.find(option) == _descriptors.end());

		// Get the option kind
		OptionKind kind = ConfigOptionType::getOptionKind<T>();

		// Add the option descriptor
		_descriptors[option] = { kind, true };

		// Initialize empty entry in default values
		_listDefaults[option] = {};

		// Set the option defaults
		for (const T &value : defaultValues) {
			_listDefaults[option].push_back(value);
		}
	}

public:
	//! \brief Initialize options and their defaults if needed
	static inline void initializeOptionsIfNeeded()
	{
		// Might trigger the initialization of config central
		getCentral();
	}

	//! \brief Get the configured value of a regular option
	//!
	//! This function first returns the value configured in the config
	//! envar or the config file, or the default value otherwise
	//!
	//! \param option The option name
	//! \param value The configured option value
	//!
	//! \returns True if the option was in the config envar/file or
	//! false otherwise
	template <typename T>
	static inline bool getOptionValue(const std::string &option, T &value)
	{
		const ConfigCentral &central = getCentral();

		// Ensure the option was registered
		const descriptors_t &descriptors = central._descriptors;
		if (descriptors.find(option) == descriptors.end())
			FatalErrorHandler::fail("Invalid config option ", option);

		// Check the config envar or file first
		ConfigParser &parser = ConfigParser::getParser();
		if (parser.get(option, value))
			return true;

		// Otherwise get the default value
		const defaults_t &defaults = central._defaults;
		auto it = defaults.find(option);
		assert(it != defaults.end());

		value = boost::get<T>(it->second);
		return false;
	}

	//! \brief Get the configured value of a list option
	//!
	//! This function first returns the value configured in the config
	//! envar or the config file, or the default value otherwise
	//!
	//! \param option The option name
	//! \param values The configured list option values
	//!
	//! \returns True if the option was in the config envar/file or
	//! false otherwise
	template <typename T>
	static inline bool getOptionValue(const std::string &option, std::vector<T> &values)
	{
		const ConfigCentral &central = getCentral();

		// Ensure the option was registered
		const descriptors_t &descriptors = central._descriptors;
		if (descriptors.find(option) == descriptors.end())
			FatalErrorHandler::fail("Invalid config option ", option);

		// Check the config envar or file first
		ConfigParser &parser = ConfigParser::getParser();
		if (parser.getList(option, values))
			return true;

		// Otherwise get the default value
		const list_defaults_t &defaults = central._listDefaults;
		auto it = defaults.find(option);
		assert(it != defaults.end());

		for (const variant_t &value : it->second) {
			values.push_back(boost::get<T>(value));
		}
		return false;
	}

	//! \brief Check whether an option exists
	//!
	//! \param option The option name
	//!
	//! \returns Whether the option exists
	static inline bool existsOption(const std::string &option)
	{
		const descriptors_t &descriptors = getCentral()._descriptors;
		auto it = descriptors.find(option);
		return (it != descriptors.end());
	}

	//! \brief Get the kind (run-time) of a valid option
	//!
	//! \param option The option name
	//!
	//! \returns The option kind
	static inline OptionKind getOptionKind(const std::string &option)
	{
		const descriptors_t &descriptors = getCentral()._descriptors;
		auto it = descriptors.find(option);
		assert(it != descriptors.end());

		return it->second._kind;
	}

	//! \brief Check whether a valid option is a list
	//!
	//! \param option The option name
	//!
	//! \returns Whether it is a list option
	static inline bool isOptionList(const std::string &option)
	{
		const descriptors_t &descriptors = getCentral()._descriptors;
		auto it = descriptors.find(option);
		assert(it != descriptors.end());

		return it->second._isList;
	}
};

// Memory size options are treated in a special way
template <>
inline bool ConfigCentral::getOptionValue(const std::string &option, memory_t &value)
{
	const ConfigCentral &central = getCentral();

	// Ensure the option was registered
	const descriptors_t &descriptors = central._descriptors;
	if (descriptors.find(option) == descriptors.end())
		FatalErrorHandler::fail("Invalid config option ", option);

	// Memory size options are processed as strings by the parser
	std::string stringified;
	ConfigParser &parser = ConfigParser::getParser();
	if (parser.get(option, stringified)) {
		// Automatic conversion
		value = stringified;
		return true;
	}

	// Otherwise get the default value
	const defaults_t &defaults = central._defaults;
	auto it = defaults.find(option);
	assert(it != defaults.end());

	value = boost::get<memory_t>(it->second);
	return false;
}

template <>
inline bool ConfigCentral::getOptionValue(const std::string &option, std::vector<memory_t> &values)
{
	const ConfigCentral &central = getCentral();

	// Ensure the option was registered
	const descriptors_t &descriptors = central._descriptors;
	if (descriptors.find(option) == descriptors.end())
		FatalErrorHandler::fail("Invalid config option ", option);

	// Memory size options are processed as strings by the parser
	std::vector<std::string> stringifieds;
	ConfigParser &parser = ConfigParser::getParser();
	if (parser.getList(option, stringifieds)) {
		for (const std::string &value : stringifieds) {
			// Automatic conversion
			values.push_back(value);
		}
		return true;
	}

	// Otherwise get the default value
	const list_defaults_t &defaults = central._listDefaults;
	auto it = defaults.find(option);
	assert(it != defaults.end());

	for (const variant_t &value : it->second) {
		values.push_back(boost::get<memory_t>(value));
	}
	return false;
}


#endif // CONFIG_CENTRAL_HPP
