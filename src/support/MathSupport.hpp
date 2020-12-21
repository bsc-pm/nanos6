/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/


#ifndef MATH_SUPPORT_HPP
#define MATH_SUPPORT_HPP

#include <cstddef>


class MathSupport {
public:
	static inline size_t ceil(size_t x, size_t y)
	{
		return (x + (y - 1)) / y;
	}

	static inline size_t closestMultiple(size_t n, size_t multipleOf)
	{
		return ((n + multipleOf - 1) / multipleOf) * multipleOf;
	}
};


#endif // MATH_SUPPORT_HPP

