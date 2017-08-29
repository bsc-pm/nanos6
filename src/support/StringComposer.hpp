/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/


#ifndef STRING_COMPOSER_HPP
#define STRING_COMPOSER_HPP


#include <string>
#include <sstream>


class StringComposer {
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
	template<typename... TS>
	static inline std::string compose(TS... contents)
	{
		std::ostringstream stream;
		fillStream(stream, contents...);
		return stream.str();
	}
};


#endif // STRING_COMPOSER_HPP

