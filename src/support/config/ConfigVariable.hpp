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
#include <unordered_set>

#include "ConfigParser.hpp"


template <typename T>
class BaseConfigVariable {
protected:
	T _value;
	bool _isPresent;
	std::string _name;

public:
	BaseConfigVariable()
	{
	}

	//! \brief Indicate if the config variable has actually been defined
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
	//! Note that this method does not alter the actual config variable. It
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

//! \brief A class to read config variables
template <typename T>
class ConfigVariable : public BaseConfigVariable<T> {
public:
	//! \brief Constructor
	//!
	//! \param[in] name the name of the config variable
	//! \param[in] defaultValue an optional value to assign if the config variable has not been defined
	ConfigVariable(const std::string &name, T defaultValue = T())
	{
		this->_name = name;
		this->_value = defaultValue;
		ConfigParser &parser = ConfigParser::getParser();
		// Does not touch the value if the variable is invalid or is not specified
		parser.get(name, this->_value, this->_isPresent);
	}
};

template <>
class ConfigVariable<StringifiedMemorySize> : public BaseConfigVariable<StringifiedMemorySize> {
public:
	//! \brief Constructor
	//!
	//! \param[in] name the name of the config variable
	//! \param[in] defaultValue an optional value to assign if the config variable has not been defined
	ConfigVariable(const std::string &name, StringifiedMemorySize defaultValue = StringifiedMemorySize())
	{
		this->_name = name;
		this->_value = defaultValue;
		ConfigParser &parser = ConfigParser::getParser();
		std::string unparsedValue;
		parser.get(name, unparsedValue, this->_isPresent);

		if (this->_isPresent)
			this->_value = StringSupport::parseMemory(unparsedValue);
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
	//! \brief Constructor
	//!
	//! \param[in] name the name of the config variable
	//! \param[in] defaultValues an optional list of values to assign if the config variable has not been defined
	ConfigVariableList(const std::string &name, std::initializer_list<T> defaultValues) :
		_contents(defaultValues),
		_name(name)
	{
		ConfigParser &parser = ConfigParser::getParser();
		// Does not touch the value if the variable is invalid or is not specified
		parser.getList(name, _contents, _isPresent);
	}

	//! \brief Indicates if the config variable has actually been defined
	inline bool isPresent() const
	{
		return _isPresent;
	}

	inline iterator begin()
	{
		return _contents.begin();
	}

	inline const_iterator begin() const
	{
		return _contents.begin();
	}

	inline iterator end()
	{
		return _contents.end();
	}

	inline const_iterator end() const
	{
		return _contents.end();
	}
};

template <typename T>
class ConfigVariableSet {
public:
	typedef std::unordered_set<T> contents_t;

private:
	contents_t _contents;
	std::string _name;
	bool _isPresent;

public:
	//! \brief Constructor
	//!
	//! \param[in] name the name of the config variable
	//! \param[in] defaultValues an optional list of values to assign if the config variable has not been defined
	ConfigVariableSet(const std::string &name, std::initializer_list<T> defaultValues) :
		_contents(defaultValues),
		_name(name)
	{
		ConfigParser &parser = ConfigParser::getParser();
		// Does not touch the value if the variable is invalid or is not specified
		std::vector<T> listContents;
		parser.getList(name, listContents, _isPresent);

		if (_isPresent) {
			_contents.clear();
			for (T &item : listContents)
				_contents.emplace(item);
		}
	}

	//! \brief Indicates if the config variable has actually been defined
	inline bool isPresent() const
	{
		return _isPresent;
	}

	inline bool contains(T &item) const
	{
		return (_contents.find(item) != _contents.end());
	}
};


#endif // CONFIG_VARIABLE_HPP
