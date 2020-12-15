/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CONFIG_CENTRAL_HPP
#define CONFIG_CENTRAL_HPP


#include <string>
#include <unordered_map>
#include <vector>

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

	//! Map used for storing regular options' default values
	template <typename T>
	using ConfigMap = std::unordered_map<std::string, T>;

	//! Map used for storing list options' default values
	template <typename T>
	using ConfigListMap = ConfigMap<std::vector<T>>;

	//! Structure used for storing option descriptors
	typedef ConfigMap<ConfigDescriptor> option_descriptors_t;

	//! Whether the options and their defaults were registered
	bool _initialized;

	//! Descriptors of valid options
	option_descriptors_t _optionDescriptors;


private:
	//! \brief Private constructor of config central
	inline ConfigCentral() :
		_initialized(false),
		_optionDescriptors()
	{
	}

	//! \brief Initialize options and their defaults
	static void initialize();

	//! \brief Get the config central
	//!
	//! The first call creates the config central object. All config
	//! central class members need to be created dynamically since this
	//! class is accessed by static config variables. Having these class
	//! members as regular static fields in the class would produce the
	//! well-known static order fiasco
	//!
	//! \returns The config central
	static inline ConfigCentral &getConfigCentral()
	{
		static ConfigCentral configCentral;
		return configCentral;
	}

	//! \brief Get the option defaults for a given type
	//!
	//! All config central class members need to be created dynamically
	//! since this class is accessed by static config variables. Having
	//! these class members as regular static fields in the class would
	//! produce the well-known static order fiasco
	//!
	//! \returns The option defaults for a given type
	template <typename T>
	static inline ConfigMap<T> &getDefaults()
	{
		static ConfigMap<T> defaults;
		return defaults;
	}

	//! \brief Get the list option defaults for a given type
	//!
	//! All config central class members need to be created dynamically
	//! since this class is accessed by static config variables. Having
	//! these class members as regular static fields in the class would
	//! produce the well-known static order fiasco
	//!
	//! \returns The list option defaults for a given type
	template <typename T>
	static inline ConfigListMap<T> &getListDefaults()
	{
		static ConfigListMap<T> defaults;
		return defaults;
	}

	//! \brief Get the option descriptors
	//!
	//! This function will create the config central object if was not
	//! already created
	//!
	//! \returns The option descriptors
	static inline option_descriptors_t &getDescriptors()
	{
		return getConfigCentral()._optionDescriptors;
	}

	//! \brief Register regular option of an accepted type
	//!
	//! This function should be called only during initialization and is
	//! not thread safe
	//!
	//! \param option The option name
	//! \param defaultValue The default value of the option
	template <typename T>
	static inline void registerOption(const std::string &option, const T &defaultValue)
	{
		assert(!existsOption(option));

		// Get the option kind
		OptionKind kind = ConfigOptionType::getOptionKind<T>();

		// Add the option descriptor
		option_descriptors_t &descriptors = getDescriptors();
		descriptors[option] = { kind, false };

		// Set the option default
		ConfigMap<T> &defaults = getDefaults<T>();
		defaults[option] = defaultValue;
	}

	//! \brief Register list option of an accepted type
	//!
	//! This function should be called only during initialization and is
	//! not thread safe
	//!
	//! \param option The option name
	//! \param defaultValues The default list of values
	template <typename T>
	static inline void registerOption(
		const std::string &option,
		const std::initializer_list<T> &defaultValues
	) {
		assert(!existsOption(option));

		// Get the option kind
		OptionKind kind = ConfigOptionType::getOptionKind<T>();

		// Add the option descriptor
		option_descriptors_t &descriptors = getDescriptors();
		descriptors[option] = { kind, true };

		// Set the option default
		ConfigListMap<T> &defaults = getListDefaults<T>();
		defaults[option] = defaultValues;
	}

	//! \brief Update regular option of an accepted type
	//!
	//! This function should be called only when initializing the default
	//! values that depend on other runtime modules, e.g. after the hardware
	//! info initialization. This function is not thread safe
	//!
	//! \param option The option name
	//! \param defaultValue The new default value of the option
	template <typename T>
	static inline void updateOption(const std::string &option, const T &defaultValue)
	{
		assert(existsOption(option));

		// Get the option kind
		OptionKind kind = ConfigOptionType::getOptionKind<T>();

		// Check that it is the same kind as the registered one
		OptionKind registeredKind = getOptionKind(option);
		if (kind != registeredKind)
			FatalErrorHandler::fail("Invalid option kind for the config option ", option);

		// Update the option default
		ConfigMap<T> &defaults = getDefaults<T>();
		defaults[option] = defaultValue;
	}

public:
	//! \brief Initialize options and their defaults if needed
	//!
	//! This function tries to register all options if it was not
	//! done previously. Usually the initialization will be done
	//! by the static config variables that we have across the
	//! runtime code. Thus, once we arrive at the Bootstrap, we
	//! will already have initialized all options
	static inline void initializeOptionsIfNeeded()
	{
		// Might trigger the creation of config central
		ConfigCentral &configCentral = getConfigCentral();

		// Register the options if needed. The initialization should
		// not be done by multiple threads at the same time
		if (!configCentral._initialized) {
			initialize();
			configCentral._initialized = true;
		}
		assert(configCentral._initialized);
	}

	//! \brief Reinitialize the memory-dependent options
	//!
	//! This function reinitializes the default values of the options
	//! that have dynamic defaults, e.g. the options that depend on the
	//! on the system memory size
	static void initializeMemoryDependentOptions();

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
		// Initialize the options if they were not registered yet
		initializeOptionsIfNeeded();

		// Ensure the option was registered
		if (!existsOption(option))
			FatalErrorHandler::fail("Invalid config option ", option);

		// Check the config envar or file first
		ConfigParser &parser = ConfigParser::getParser();
		if (parser.get(option, value))
			return true;

		// Otherwise get the default value
		const ConfigMap<T> &defaults = getDefaults<T>();
		auto it = defaults.find(option);
		assert(it != defaults.end());

		value = it->second;
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
		// Initialize the options if they were not registered yet
		initializeOptionsIfNeeded();

		// Ensure the option was registered
		if (!existsOption(option))
			FatalErrorHandler::fail("Invalid config option ", option);

		// Check the config envar or file first
		ConfigParser &parser = ConfigParser::getParser();
		if (parser.getList(option, values))
			return true;

		// Otherwise get the default value
		const ConfigListMap<T> &defaults = getListDefaults<T>();
		auto it = defaults.find(option);
		assert(it != defaults.end());

		values = it->second;
		return false;
	}

	//! \brief Check whether an option exists
	//!
	//! \param option The option name
	//!
	//! \returns Whether the option exists
	static inline bool existsOption(const std::string &option)
	{
		const option_descriptors_t &descriptors = getDescriptors();
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
		const option_descriptors_t &descriptors = getDescriptors();
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
		const option_descriptors_t &descriptors = getDescriptors();
		auto it = descriptors.find(option);
		assert(it != descriptors.end());

		return it->second._isList;
	}
};

// Memory size options are treated in a special way
template <>
inline bool ConfigCentral::getOptionValue(const std::string &option, memory_t &value)
{
	std::string stringified;

	// Initialize the options if they were not registered yet
	initializeOptionsIfNeeded();

	// Ensure the option was registered
	if (!existsOption(option))
		FatalErrorHandler::fail("Invalid config option ", option);

	// Memory size options are processed as strings by the parser
	ConfigParser &parser = ConfigParser::getParser();
	if (parser.get(option, stringified)) {
		// Automatic conversion
		value = stringified;
		return true;
	}

	// Otherwise get the default value
	ConfigMap<memory_t> &defaults = getDefaults<memory_t>();
	auto it = defaults.find(option);
	assert(it != defaults.end());

	value = it->second;
	return false;
}

template <>
inline bool ConfigCentral::getOptionValue(const std::string &option, std::vector<memory_t> &values)
{
	std::vector<std::string> stringifieds;

	// Initialize the options if they were not registered yet
	initializeOptionsIfNeeded();

	// Ensure the option was registered
	if (!existsOption(option))
		FatalErrorHandler::fail("Invalid config option ", option);

	// Memory size options are processed as strings by the parser
	ConfigParser &parser = ConfigParser::getParser();
	if (parser.getList(option, stringifieds)) {
		values.clear();
		for (const std::string &value : stringifieds) {
			// Automatic conversion
			values.push_back(value);
		}
		return true;
	}

	// Otherwise get the default value
	ConfigListMap<memory_t> &defaults = getListDefaults<memory_t>();
	auto it = defaults.find(option);
	assert(it != defaults.end());

	values = it->second;
	return false;
}


#endif // CONFIG_CENTRAL_HPP
