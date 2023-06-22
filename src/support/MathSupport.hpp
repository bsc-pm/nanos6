/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2023 Barcelona Supercomputing Center (BSC)
*/


#ifndef MATH_SUPPORT_HPP
#define MATH_SUPPORT_HPP

class MathSupport {
public:
	template <typename T>
	static inline T ceil(T x, T y)
	{
		return (x + (y - (T) 1)) / y;
	}

	template <typename T>
	static inline T closestMultiple(T n, T multipleOf)
	{
		return ((n + multipleOf - (T) 1) / multipleOf) * multipleOf;
	}
};


#endif // MATH_SUPPORT_HPP

