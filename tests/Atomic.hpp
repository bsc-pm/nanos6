/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef ATOMIC_HPP
#define ATOMIC_HPP


#if __cplusplus >= 201103L

#include <atomic>
template <typename T>
using Atomic = std::atomic<T>;

#elif USE_BLOCKING_API

// Unfortunately this changes behaviour of the test

// C++ 03

#include <nanos6/blocking.h>

template <typename T>
class Atomic {
	volatile T _value;
	
public:
	Atomic()
	{
	}
	
	Atomic(T const &value)
		: _value(value)
	{
	}
	
	T load()
	{
		T value;
		
		#pragma oss critical
		value = _value;
		
		return value;
	}
	
	void store(T const &value)
	{
		#pragma oss critical
		_value = value;
	}
	
	T operator++()
	{
		T value;
		#pragma oss critical
		{
			_value++;
			value = _value;
		}
		return value;
	}
	
	T operator--()
	{
		T value;
		#pragma oss critical
		{
			_value--;
			value = _value;
		}
		return value;
	}
	
	T operator++(int)
	{
		T value;
		#pragma oss critical
		{
			value = _value;
			_value++;
		}
		return value;
	}
	
	T operator--(int)
	{
		T value;
		#pragma oss critical
		{
			value = _value;
			_value--;
		}
		return value;
	}
	
	operator T()
	{
		T value;
		
		#pragma oss critical
		value = _value;
		
		return value;
	}
	
	void operator=(T const &value)
	{
		#pragma oss critical
		_value = value;
	}
	
};


#else

// C++03

#include <pthread.h>


template <typename T>
class Atomic {
	pthread_mutex_t _mutex;
	volatile T _value;
	
public:
	Atomic()
	{
		pthread_mutex_init(&_mutex, 0);
	}
	
	Atomic(T const &value)
		: _value(value)
	{
		pthread_mutex_init(&_mutex, 0);
	}
	
	~Atomic()
	{
		pthread_mutex_destroy(&_mutex);
	}
	
	T load()
	{
		volatile T value;
		
		pthread_mutex_lock(&_mutex);
		value = _value;
		pthread_mutex_unlock(&_mutex);
		
		return value;
	}
	
	void store(T const &value)
	{
		pthread_mutex_lock(&_mutex);
		_value = value;
		pthread_mutex_unlock(&_mutex);
	}
	
	T operator++()
	{
		volatile T value;
		pthread_mutex_lock(&_mutex);
		{
			_value++;
			value = _value;
		}
		pthread_mutex_unlock(&_mutex);
		return value;
	}
	
	T operator--()
	{
		volatile T value;
		pthread_mutex_lock(&_mutex);
		{
			_value--;
			value = _value;
		}
		pthread_mutex_unlock(&_mutex);
		return value;
	}
	
	T operator++(int)
	{
		volatile T value;
		pthread_mutex_lock(&_mutex);
		{
			value = _value;
			_value++;
		}
		pthread_mutex_unlock(&_mutex);
		return value;
	}
	
	T operator--(int)
	{
		volatile T value;
		pthread_mutex_lock(&_mutex);
		{
			value = _value;
			_value--;
		}
		pthread_mutex_unlock(&_mutex);
		return value;
	}
	
	operator T()
	{
		volatile T value;
		
		pthread_mutex_lock(&_mutex);
		value = _value;
		pthread_mutex_unlock(&_mutex);
		
		return value;
	}
	
	void operator=(T const &value)
	{
		pthread_mutex_lock(&_mutex);
		_value = value;
		pthread_mutex_unlock(&_mutex);
	}
	
};

#endif


#endif // ATOMIC_HPP
