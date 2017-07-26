/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TOKENIZED_ENVIORNMENT_VARIABLE_HPP
#define TOKENIZED_ENVIORNMENT_VARIABLE_HPP


#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


//! \brief A class to read environment variables that contain lists
template <typename T>
class TokenizedEnvironmentVariable {
public:
	typedef std::vector<T> contents_t;
	typedef typename contents_t::iterator iterator;
	typedef typename contents_t::const_iterator const_iterator;
	
private:
	contents_t _contents;
	bool _isPresent;
	std::string _name;
	
public:
	//! \brief constructor
	//!
	//! \param[in] name the name of the environment variable
	//! \param[in] defaultValue an optional value to assign if the environment variable has not been defined
	TokenizedEnvironmentVariable(std::string const &name, char delimiter, std::string const &defaultValues)
		: _name(name)
	{
		char const *contentsString = getenv(name.c_str());
		if (contentsString == nullptr) {
			contentsString = defaultValues.c_str();
			_isPresent = false;
		} else {
			_isPresent = true;
		}
		
		std::istringstream iss(contentsString);
		std::string element;
		while (std::getline(iss, element, delimiter)) {
			_contents.push_back(element);
		}
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


#endif // TOKENIZED_ENVIORNMENT_VARIABLE_HPP
