/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/


#ifndef MATH_SUPPORT_HPP
#define MATH_SUPPORT_HPP

namespace MathSupport {
	static inline size_t ceil(size_t x, size_t y)
	{
		return (x+(y-1))/y;
	}
}


#endif // MATH_SUPPORT_HPP

