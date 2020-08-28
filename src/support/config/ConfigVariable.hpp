/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CONFIG_VARIABLE_HPP
#define CONFIG_VARIABLE_HPP


#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "ConfigParser.hpp"

class ConfigVariableAux {
	public:
	static inline ConfigParser &getParser()
	{
		static ConfigParser configParser;
		return configParser;
	}
};

template <typename T>
class BaseConfigVariable {
protected:
	T _value;
	bool _isPresent;
	std::string _name;

public:
	BaseConfigVariable() { }

	//! \brief indicates if the config variable has actually been defined
	inline bool isPresent() const
	{
		return _isPresent;
	}

	//! \brief retrieve the current value
	inline T getValue() const
	{
		return _value;
	}

	//! \brief retrieve the current value
	operator T() const
	{
		return _value;
	}

	//! \brief overwrite the value
	//!
	//! Note that this method does not alter the actual config variable. It
	//! only modifies the value stored in the object.
	//!
	//! \param[in] value the new value
	//! \param[in] makePresent mark it as if it had been originally defined
	inline void setValue(T value, bool makePresent=false)
	{
		_value = value;
		_isPresent |= makePresent;
	}
};

//! \brief A class to read config variables
template <typename T>
class ConfigVariable : public BaseConfigVariable<T> {
public:
	//! \brief constructor
	//!
	//! \param[in] name the name of the config variable
	//! \param[in] defaultValue an optional value to assign if the config variable has not been defined
	ConfigVariable(std::string const &name, T defaultValue = T())
	{
		this->_name = name;
		this->_value = defaultValue;
		ConfigParser &parser = ConfigVariableAux::getParser();
		// Does not touch the value if the variable is invalid or is not specified
		parser.get(name, this->_value, this->_isPresent);
	}
};

template <>
class ConfigVariable<StringifiedMemorySize> : public BaseConfigVariable<StringifiedMemorySize> {
public:
	ConfigVariable(std::string const &name, StringifiedMemorySize defaultValue = StringifiedMemorySize())
	{
		this->_name = name;
		this->_value = defaultValue;
		ConfigParser &parser = ConfigVariableAux::getParser();
		std::string unparsedValue;
		parser.get(name, unparsedValue, this->_isPresent);

		if (this->_isPresent)
			this->_value = memparse(unparsedValue);
	}

private:

	/** It parses a string representing a size in the form
	 * 'xxxx[k|K|m|M|g|G|t|T|p|P|e|E]' to a size_t value. */
	inline size_t memparse(std::string str)
	{
		char *endptr;

		size_t ret = strtoull(str.c_str(), &endptr, 0);

		switch (*endptr) {
			case 'E':
			case 'e':
				ret <<= 10;
				// fall through
			case 'P':
			case 'p':
				ret <<= 10;
				// fall through
			case 'T':
			case 't':
				ret <<= 10;
				// fall through
			case 'G':
			case 'g':
				ret <<= 10;
				// fall through
			case 'M':
			case 'm':
				ret <<= 10;
				// fall through
			case 'K':
			case 'k':
				ret <<= 10;
				endptr++;
				// fall through
			default:
				break;
		}

		return ret;
	}
};

template <typename T>
class ConfigVariableList {
public:
	typedef std::vector<T> contents_t;
	typedef typename contents_t::iterator iterator;
	typedef typename contents_t::const_iterator const_iterator;

private:
	contents_t _contents;
	std::string _name;
	bool _isPresent;

public:
	ConfigVariableList(std::string const &name, std::initializer_list<T> defaultValues) :
		_contents(defaultValues), _name(name)
	{
		ConfigParser &parser = ConfigVariableAux::getParser();
		// Does not touch the value if the variable is invalid or is not specified
		parser.getList(name, _contents, _isPresent);
	}

	//! \brief indicates if the enviornment variable has actually been defined
	inline bool isPresent() const
	{
		return _isPresent;
	}

	iterator begin()
	{
		return _contents.begin();
	}
	const_iterator begin() const
	{
		return _contents.begin();
	}

	iterator end()
	{
		return _contents.end();
	}
	const_iterator end() const
	{
		return _contents.end();
	}

};


#endif // CONFIG_VARIABLE_HPP
