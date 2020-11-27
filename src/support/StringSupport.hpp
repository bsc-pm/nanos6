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

	StringifiedMemorySize(const StringifiedMemorySize &other) :
		_value(other._value)
	{
	}

	StringifiedMemorySize(const std::string &value);

	StringifiedMemorySize& operator=(const StringifiedMemorySize& other) = default;

	void operator=(const std::string &value);

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

	//! \brief Find an assignment or compare operator
	//!
	//! \param str The string where to search
	//! \param isCondition Whether should search a compare operator
	//! \param op The found operator
	//!
	//! \returns The index of the operator in the string or
	//! std::string::npos if not found
	static inline size_t findOperator(const std::string &str, bool isCondition, std::string &op)
	{
		if (isCondition) {
			constexpr size_t numOps = 6;
			const char *condOperators[numOps] = { "==", "!=", ">=", "<=", ">", "<" };

			// Check always the longest operators first
			for (size_t i = 0; i < numOps; ++i) {
				size_t index = str.find(condOperators[i]);
				if (index != std::string::npos) {
					op = condOperators[i];
					return index;
				}
			}
		} else {
			const char *assigOperator = "=";
			size_t index = str.find(assigOperator);
			if (index != std::string::npos) {
				op = assigOperator;
				return index;
			}
		}
		return std::string::npos;
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
	static inline bool parse(const char *str, T &result, io_manipulator_t manipulator = std::noboolalpha)
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

inline StringifiedMemorySize::StringifiedMemorySize(const std::string &value)
{
	_value = StringSupport::parseMemory(value);
}

inline void StringifiedMemorySize::operator=(const std::string &value)
{
	_value = StringSupport::parseMemory(value);
}

template <>
inline bool StringSupport::parse(const char *str, std::string &result, io_manipulator_t)
{
	result = str;
	return true;
}

template <>
inline bool StringSupport::parse(const char *str, StringifiedMemorySize &result, io_manipulator_t)
{
	result = parseMemory(str);
	return true;
}


#endif // STRING_SUPPORT_HPP
