/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/


#ifndef OBJECTIFIED_HPP
#define OBJECTIFIED_HPP


template <typename T, T DEFAULT_VALUE = ~0UL>
class Objectified {
	T _value;
	
public:
	Objectified()
		: _value(DEFAULT_VALUE)
	{
	}
	
	Objectified(T value)
	: _value(value)
	{
	}
	
	operator T() const
	{
		return _value;
	}
	
	Objectified &operator++()
	{
		++_value;
		return *this;
	}
	
	Objectified operator++(int)
	{
		T result = _value;
		_value++;
		return Objectified(result);
	}
	
	Objectified &operator--()
	{
		--_value;
		return *this;
	}
	
	Objectified operator--(int)
	{
		T result = _value;
		_value--;
		return Objectified(result);
	}
};


#endif // OBJECTIFIED_HPP
