#ifndef ENVIORNMENT_VARIABLE_HPP
#define ENVIORNMENT_VARIABLE_HPP


#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>


//! \brief A class to read environment variables
template <typename T>
class EnvironmentVariable {
private:
	T _value;
	bool _isPresent;
	std::string _name;
	
public:
	//! \brief constructor
	//!
	//! \param[in] name the name of the environment variable
	//! \param[in] defaultValue an optional value to assign if the environment variable has not been defined
	EnvironmentVariable(std::string const &name, T defaultValue = T())
		: _value(defaultValue), _name(name)
	{
		char const *valueString = getenv(name.c_str());
		if (valueString != nullptr) {
			std::istringstream iss(valueString);
			T assignedValue = defaultValue;
			iss >> assignedValue;
			if (!iss.fail()) {
				_value = assignedValue;
				_isPresent = true;
			} else {
				std::cerr << "Warning: invalid value for environment variable " << name << ". Defaulting to " << defaultValue << "." << std::endl;
				_isPresent = false;
			}
		} else {
			_isPresent = false;
		}
	}
	
	//! \brief indicates if the enviornment variable has actually been defined
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
	//! Note that this method does not alter the actual enviornment variable. It
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


#endif // ENVIORNMENT_VARIABLE_HPP
