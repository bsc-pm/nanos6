/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2017-2020 Barcelona Supercomputing Center (BSC)
*/


#ifndef STRING_SUPPORT_HPP
#define STRING_SUPPORT_HPP


#include <string>
#include <sstream>


class StringifiedBool {
private:
	bool _value;

public:
	StringifiedBool() :
		_value(false)
	{
	}

	StringifiedBool(bool value) :
		_value(value)
	{
	}

	StringifiedBool(StringifiedBool const &other) :
		_value(other._value)
	{
	}

	StringifiedBool& operator=(const StringifiedBool& other) = default;

	operator bool() const
	{
		return _value;
	}
};


class StringifiedMemorySize {
private:
	size_t _value;

public:
	StringifiedMemorySize() :
		_value(0)
	{
	}

	StringifiedMemorySize(size_t value) :
		_value(value)
	{
	}

	StringifiedMemorySize(StringifiedMemorySize const &other) :
		_value(other._value)
	{
	}

	StringifiedMemorySize& operator=(const StringifiedMemorySize& other) = default;

	operator size_t() const
	{
		return _value;
	}
};


class StringSupport {
private:
	template<typename T>
	static inline void fillStream(std::ostringstream &stream, T contents)
	{
		stream << contents;
	}

	template<typename T, typename... TS>
	static inline void fillStream(std::ostringstream &stream, T content1, TS... contents)
	{
		stream << content1;
		fillStream(stream, contents...);
	}

public:
	typedef std::ios_base& (*io_manipulator_t)(std::ios_base&);

	//! \brief Compose a string
	//!
	//! \param[in] contents the contents used to compose the string
	//!
	//! \returns the composed string
	template<typename... TS>
	static inline std::string compose(TS... contents)
	{
		std::ostringstream stream;
		fillStream(stream, contents...);
		return stream.str();
	}

	//! \brief Parse string representing a type
	//!
	//! By default this function parses boolean values in the form of 0/1, but
	//! can be modified by passing the std::boolalpha manipulator to parse them
	//! as true/false. The parsing of strings always ignore the manipulator
	//!
	//! \param[in] str the string to parse
	//! \param[out] result the place where to store the parsed value if succeeds
	//! \param[in] manipulator the IO manipulator to apply
	//!
	//! \returns whether the parsing succeeded
	template <typename T>
	static inline bool parse(const std::string &str, T &result, io_manipulator_t manipulator = std::noboolalpha)
	{
		return parse(str.c_str(), result, manipulator);
	}

	//! \brief Parse string representing a type
	//!
	//! By default this function parses boolean values in the form of 0/1, but
	//! can be modified by passing the std::boolalpha manipulator to parse them
	//! as true/false. The parsing of strings always ignore the manipulator
	//!
	//! \param[in] str the string to parse
	//! \param[out] result the place where to store the parsed value if succeeds
	//! \param[in] manipulator the IO manipulator to apply
	//!
	//! \returns whether the parsing succeeded
	template <typename T>
	static bool parse(const char *str, T &result, io_manipulator_t manipulator = std::noboolalpha);

	//! \brief Parse string representing memory size
	//!
	//! This function parses a string representing a size in
	//! the form 'xxxx[k|K|m|M|g|G|t|T|p|P|e|E]' to a size_t
	//!
	//! \param[in] str the string to parse
	//!
	//! \returns the parsed memory size
	static inline size_t parseMemory(const std::string &str)
	{
		return parseMemory(str.c_str());
	}

	//! \brief Parse string representing memory size
	//!
	//! This function parses a string representing a size in
	//! the form 'xxxx[k|K|m|M|g|G|t|T|p|P|e|E]' to a size_t
	//!
	//! \param[in] str the string to parse
	//!
	//! \returns the parsed memory size
	static inline size_t parseMemory(const char *str)
	{
		char *endptr;
		size_t ret = strtoull(str, &endptr, 0);

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
				// fall through
			default:
				break;
		}
		return ret;
	}
};


template <typename T>
inline bool StringSupport::parse(const char *str, T &result, io_manipulator_t manipulator)
{
	std::istringstream iss(str);
	T tmp = result;

	iss >> manipulator >> tmp;

	if (!iss.fail()) {
		result = tmp;
		return true;
	}
	return false;
}

template <>
inline bool StringSupport::parse(const char *str, std::string &result, io_manipulator_t)
{
	result = str;
	return true;
}


#endif // STRING_SUPPORT_HPP

