/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef ENVIRONMENT_VARIABLE_HPP
#define ENVIRONMENT_VARIABLE_HPP


#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "support/StringSupport.hpp"


//! \brief A class to read environment variables
template <typename T>
class EnvironmentVariable {
private:
	T _value;
	bool _isPresent;
	std::string _name;

public:
	//! \brief Constructor
	//!
	//! \param[in] name the name of the environment variable
	//! \param[in] defaultValue an optional value to assign if the environment variable has not been defined
	EnvironmentVariable(const std::string &name, T defaultValue = T()) :
		_value(defaultValue),
		_name(name)
	{
		char const *value = getenv(name.c_str());
		if (value != nullptr) {
			if (StringSupport::parse<T>(value, _value)) {
				_isPresent = true;
			} else {
				std::cerr << "Warning: invalid value for environment variable " << name << ". Defaulting to " << defaultValue << "." << std::endl;
				_isPresent = false;
			}
		} else {
			_isPresent = false;
		}
	}

	//! \brief Indicate if the enviornment variable has actually been defined
	inline bool isPresent() const
	{
		return _isPresent;
	}

	//! \brief Retrieve the current value
	inline T getValue() const
	{
		return _value;
	}

	//! \brief Retrieve the current value
	operator T() const
	{
		return _value;
	}

	//! \brief Overwrite the value
	//!
	//! Note that this method does not alter the actual enviornment variable. It
	//! only modifies the value stored in the object.
	//!
	//! \param[in] value the new value
	//! \param[in] makePresent mark it as if it had been originally defined
	inline void setValue(T value, bool makePresent = false)
	{
		_value = value;
		_isPresent |= makePresent;
	}
};


template <>
class EnvironmentVariable<StringifiedMemorySize> {
private:
	StringifiedMemorySize _value;
	bool _isPresent;
	std::string _name;

public:
	EnvironmentVariable(const std::string &name, StringifiedMemorySize defaultValue = StringifiedMemorySize()) :
		_value(defaultValue),
		_name(name)
	{
		char const *valueString = getenv(name.c_str());
		if (valueString != nullptr) {
			_value = StringSupport::parseMemory(valueString);
			_isPresent = true;
		} else {
			_isPresent = false;
		}
	}

	inline bool isPresent() const
	{
		return _isPresent;
	}

	inline StringifiedMemorySize getValue() const
	{
		return _value;
	}

	operator StringifiedMemorySize() const
	{
		return _value;
	}

	inline void setValue(StringifiedMemorySize value, bool makePresent = false)
	{
		_value = value;
		_isPresent |= makePresent;
	}
};


#endif // ENVIRONMENT_VARIABLE_HPP
