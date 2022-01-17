
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

template class ConfigVariable<std::string>;
template class ConfigVariable<StringifiedMemorySize>;
template class ConfigVariable<int>;
template class ConfigVariable<bool>;
template class ConfigVariable<uint64_t>;

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

template class ConfigVariableSet<std::string>;
