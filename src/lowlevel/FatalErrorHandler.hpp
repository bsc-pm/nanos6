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

#include <string.h>
#include <unistd.h>


class FatalErrorHandler {
protected:
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
	
	static inline void safeEmitPart(__attribute__((unused)) char *buffer, __attribute__((unused)) size_t size, char part)
	{
		write(2, &part, 1);
	}
	
	static inline void safeEmitPart(char *buffer, size_t size, int part)
	{
		int length = snprintf(buffer, size, "%i", part);
		write(2, buffer, length);
	}
	
	static inline void safeEmitPart(char *buffer, size_t size, long part)
	{
		int length = snprintf(buffer, size, "%li", part);
		write(2, buffer, length);
	}
	
	static inline void safeEmitPart(__attribute__((unused)) char *buffer, __attribute__((unused)) size_t size, char const *part)
	{
		write(2, part, strlen(part));
	}
	
	static inline void safeEmitPart(char *buffer, size_t size, float part)
	{
		int length = snprintf(buffer, size, "%f", part);
		write(2, buffer, length);
	}
	
	static inline void safeEmitPart(char *buffer, size_t size, double part)
	{
		int length = snprintf(buffer, size, "%f", part);
		write(2, buffer, length);
	}
	
	template<typename T, typename... TS>
	static inline void safeEmitReasonParts(char *buffer, size_t size, T const &firstReasonPart, TS... reasonParts)
	{
		safeEmitPart(buffer, size, firstReasonPart);
		safeEmitReasonParts(buffer, size, reasonParts...);
	}
	
	static inline void safeEmitReasonParts(__attribute__((unused)) char *buffer, __attribute__((unused)) size_t size)
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
	static inline void safeHandle(int rc, char *buffer, size_t size, TS... reasonParts)
	{
		if (__builtin_expect(rc == 0, 1)) {
			return;
		}
		
		{
			std::lock_guard<SpinLock> guard(_lock);
			write(2, "Error: ", 7);
			strerror_r(rc, buffer, size);
			write(2, buffer, strlen(buffer));
			safeEmitReasonParts(buffer, size, reasonParts...);
			write(2, "\n", strlen("\n"));
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
	
	template<typename... TS>
	static inline void warnIf(bool failure, TS... reasonParts)
	{
		if (__builtin_expect(!failure, 1)) {
			return;
		}
		
		std::ostringstream oss;
		oss << "Warning: ";
		emitReasonParts(oss, reasonParts...);
		oss << std::endl;
		
		{
			std::lock_guard<SpinLock> guard(_lock);
			std::cerr << oss.str();
		}
	}
	
};


#endif // FATAL_ERROR_HANDLER_HPP
