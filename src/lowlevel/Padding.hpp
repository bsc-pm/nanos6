/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2019-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef PADDING_HPP
#define PADDING_HPP

#include <cstddef>
#include <config.h>

// CACHELINE_SIZE is defined by the Nanos6 configure step.
#ifndef CACHELINE_SIZE
#warning "Cacheline size is not specified. Falling back to safe default."
#define CACHELINE_SIZE 128
#endif

template<class T, size_t Size = CACHELINE_SIZE>
class Padded : public T {
	using T::T;

	constexpr static size_t roundup(size_t const x, size_t const y)
	{
		return (((x + (y - 1)) / y) * y);
	}

	uint8_t padding[roundup(sizeof(T), Size)-sizeof(T)];

public:
	inline T *ptr_to_basetype()
	{
		return (T *) this;
	}
};

#endif // PADDING_HPP
