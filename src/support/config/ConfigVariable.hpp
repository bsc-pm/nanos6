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

#include "ConfigCentral.hpp"
#include "ConfigParser.hpp"


//! Class representing a configuration variable
template <typename T>
class ConfigVariable {
private:
	typedef typename ConfigOptionType::type<T> option_type_t;

	std::string _name;
	T _value;
	bool _isPresent;

public:
	//! \brief Constructor
	//!
	//! \param[in] name the name of the config variable
	ConfigVariable(const std::string &name) :
		_name(name),
		_value(),
		_isPresent(false)
	{
		option_type_t value;
		_isPresent = ConfigCentral::getOptionValue<option_type_t>(_name, value);
		_value = (T) value;
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
	//! only modifies the value stored in the object
	//!
	//! \param[in] value the new value
	inline void setValue(T value)
	{
		_value = value;
	}
};


//! Class representing a configuration variable set
template <typename T>
class ConfigVariableSet {
public:
	typedef std::unordered_set<T> contents_t;
	typedef typename contents_t::iterator iterator;
	typedef typename contents_t::const_iterator const_iterator;

private:
	typedef typename ConfigOptionType::type<T> option_type_t;

	std::string _name;
	contents_t _contents;
	bool _isPresent;

public:
	//! \brief Constructor
	//!
	//! \param[in] name The name of the config variable
	ConfigVariableSet(const std::string &name) :
		_name(name),
		_contents(),
		_isPresent(false)
	{
		std::vector<option_type_t> values;
		_isPresent = ConfigCentral::getOptionValue<option_type_t>(_name, values);

		if (!values.empty()) {
			_contents.clear();
			for (T &item : values)
				_contents.emplace((T) item);
		}
	}

	//! \brief Indicate if the config variable has actually been defined
	inline bool isPresent() const
	{
		return _isPresent;
	}

	//! \brief Indicate if the config variable contains an item
	//!
	//! \param item The item to search
	inline bool contains(T &item) const
	{
		return (_contents.find(item) != _contents.end());
	}

	//! \brief Retrieve an iterator to the beginning
	inline iterator begin()
	{
		return _contents.begin();
	}

	//! \brief Retrieve an iterator to the beginning
	inline const_iterator begin() const
	{
		return _contents.begin();
	}

	//! \brief Retrieve an iterator to the end
	inline iterator end()
	{
		return _contents.end();
	}

	//! \brief Retrieve an iterator to the end
	inline const_iterator end() const
	{
		return _contents.end();
	}
};


#endif // CONFIG_VARIABLE_HPP
