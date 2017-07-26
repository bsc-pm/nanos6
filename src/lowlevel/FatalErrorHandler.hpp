/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef FATAL_ERROR_HANDLER_HPP
#define FATAL_ERROR_HANDLER_HPP

#include <string.h>

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>

#include "lowlevel/SpinLock.hpp"


class FatalErrorHandler {
private:
	static SpinLock _lock;
	
	template<typename T, typename... TS>
	static inline void emitReasonParts(std::ostringstream &oss, T const &firstReasonPart, TS... reasonParts)
	{
		oss << firstReasonPart;
		emitReasonParts(oss, reasonParts...);
	}
	
	static inline void emitReasonParts(__attribute__((unused)) std::ostringstream &oss)
	{
	}
	
public:
	template<typename... TS>
	static inline void handle(int rc, TS... reasonParts)
	{
		if (__builtin_expect(rc == 0, 1)) {
			return;
		}
		
		std::ostringstream oss;
		oss << "Error: " << strerror(rc);
		emitReasonParts(oss, reasonParts...);
		oss << std::endl;
		
		{
			std::lock_guard<SpinLock> guard(_lock);
			std::cerr << oss.str();
		}
		
#ifndef NDEBUG
		abort();
#else
		exit(1);
#endif
	}
	
	
	template<typename... TS>
	static inline void check(bool success, TS... reasonParts)
	{
		if (__builtin_expect(success, 1)) {
			return;
		}
		
		std::ostringstream oss;
		oss << "Error: ";
		emitReasonParts(oss, reasonParts...);
		oss << std::endl;
		
		{
			std::lock_guard<SpinLock> guard(_lock);
			std::cerr << oss.str();
		}
		
#ifndef NDEBUG
		abort();
#else
		exit(1);
#endif
	}
	
	template<typename... TS>
	static inline void failIf(bool failure, TS... reasonParts)
	{
		if (__builtin_expect(!failure, 1)) {
			return;
		}
		
		std::ostringstream oss;
		oss << "Error: ";
		emitReasonParts(oss, reasonParts...);
		oss << std::endl;
		
		{
			std::lock_guard<SpinLock> guard(_lock);
			std::cerr << oss.str();
		}
		
#ifndef NDEBUG
		abort();
#else
		exit(1);
#endif
	}
	
};


#endif // FATAL_ERROR_HANDLER_HPP
