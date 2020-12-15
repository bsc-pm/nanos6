/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef CONFIG_SUPPORT_HPP
#define CONFIG_SUPPORT_HPP

#include <cstdint>
#include <string>
#include <type_traits>

#include "support/StringSupport.hpp"


//! Config option's accepted types (compile-time) and kinds (run-time)
class ConfigOptionType {
public:
	//! Option kinds (run-time)
	enum OptionKind {
		bool_option = 0,
		float_option,
		integer_option,
		string_option,
		memory_option,
		num_options
	};

	//! Accepted option types (compile-time)
	typedef bool bool_t;
	typedef double float_t;
	typedef uint64_t integer_t;
	typedef std::string string_t;
	typedef StringifiedMemorySize memory_t;

private:
	//! Utils for templates targeting floating point
	template <typename T>
	using IsFloat = typename std::enable_if<std::is_floating_point<T>::value>::type;

	//! Utils for templates targeting integers (non-booleans)
	template <typename T>
	using IsInteger = typename std::enable_if<std::is_integral<T>::value && !std::is_same<T, bool>::value>::type;

	//! Config type for all classes and types except floats and integers
	template <class T, class Enable = void>
	class Type {
	public:
		typedef T type;
	};

public:
	//! Type to accepted config type converter (compile-time)
	template <typename T>
	using type = typename Type<T>::type;

	//! \brief Get the option kind for a given type (run-time)
	//!
	//! The template type should be one of the accepted option types
	//! above. Calling this function with any unrecognized option type
	//! will produce a compilation error
	//!
	//! \returns The corresponding option kind
	template <typename T>
	static inline OptionKind getOptionKind()
	{
		// We only allow the accepted types defined above
		static_assert(sizeof(T) < 0, "This is not accepted config option type");
		return num_options;
	}
};

//! Config option type for floating point
template <class T>
class ConfigOptionType::Type<T, ConfigOptionType::IsFloat<T>> {
public:
	typedef ConfigOptionType::float_t type;
};

//! Config option type for integers (non-boolean)
template <class T>
class ConfigOptionType::Type<T, ConfigOptionType::IsInteger<T>> {
public:
	typedef ConfigOptionType::integer_t type;
};

//! Accepted config type specializations
template <>
inline ConfigOptionType::OptionKind ConfigOptionType::getOptionKind<ConfigOptionType::bool_t>()
{
	return OptionKind::bool_option;
}

template <>
inline ConfigOptionType::OptionKind ConfigOptionType::getOptionKind<ConfigOptionType::float_t>()
{
	return OptionKind::float_option;
}

template <>
inline ConfigOptionType::OptionKind ConfigOptionType::getOptionKind<ConfigOptionType::integer_t>()
{
	return OptionKind::integer_option;
}

template <>
inline ConfigOptionType::OptionKind ConfigOptionType::getOptionKind<ConfigOptionType::string_t>()
{
	return OptionKind::string_option;
}

template <>
inline ConfigOptionType::OptionKind ConfigOptionType::getOptionKind<ConfigOptionType::memory_t>()
{
	return OptionKind::memory_option;
}

#endif // CONFIG_SUPPORT_HPP
