/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#include "ConfigVariable.hpp"

#include <support/StringSupport.hpp>
#include <support/config/ConfigSupport.hpp>

#include "ConfigCentral.hpp"
#include "ConfigParser.hpp"


template <typename T>
ConfigVariable<T>::ConfigVariable(const std::string &name) :
	_name(name),
	_value(),
	_isPresent(false)
{
	typedef typename ConfigOptionType::type<T> option_type_t;

	option_type_t value;
	_isPresent = ConfigCentral::getOptionValue<option_type_t>(_name, value);
	_value = (T) value;
}

template <typename T>
ConfigVariableSet<T>::ConfigVariableSet(const std::string &name) :
	_name(name),
	_contents(),
	_isPresent(false)
{
	typedef typename ConfigOptionType::type<T> option_type_t;

	std::vector<option_type_t> values;
	_isPresent = ConfigCentral::getOptionValue<option_type_t>(_name, values);

	if (!values.empty()) {
		_contents.clear();
		for (T &item : values)
			_contents.emplace((T) item);
	}
}

//! ConfigVariable and ConfigVariableSet types must be declared here. They
//! are templates but we define them in this source file to cut compilation
//! times. Processing the header of ConfigCentral and toml is very expensive

template class ConfigVariable<bool>;
template class ConfigVariable<int>;
template class ConfigVariable<size_t>;
template class ConfigVariable<unsigned int>;
template class ConfigVariable<std::string>;
template class ConfigVariable<StringifiedMemorySize>;

template class ConfigVariableSet<std::string>;
