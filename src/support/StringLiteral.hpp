/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef STRING_LITERAL_HPP
#define STRING_LITERAL_HPP


class StringLiteral {
private:
	char const *_value;
	
public:
	StringLiteral(char const *value)
		: _value(value)
	{
	}
	
	operator char const * () const
	{
		return _value;
	}
};


#endif // STRING_LITERAL_HPP
