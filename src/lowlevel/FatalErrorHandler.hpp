#ifndef FATAL_ERROR_HANDLER_HPP
#define FATAL_ERROR_HANDLER_HPP

#include <string.h>

#include <cstdlib>
#include <iostream>
#include <mutex>

#include "lowlevel/SpinLock.hpp"


class FatalErrorHandler {
private:
	static SpinLock _lock;
	
	template<typename T, typename... TS>
	static inline void emitReasonParts(T const &firstReasonPart, TS... reasonParts)
	{
		std::cout << firstReasonPart;
		emitReasonParts(reasonParts...);
	}
	
	static inline void emitReasonParts()
	{
	}
	
public:
	template<typename... TS>
	static inline void handle(int rc, TS... reasonParts)
	{
		if (__builtin_expect(rc == 0, 1)) {
			return;
		}
		
		std::lock_guard<SpinLock> guard(_lock);
		
		std::cerr << "Error: " << strerror(rc);
		emitReasonParts(reasonParts...);
		std::cerr << std::endl;
		
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
		
		std::lock_guard<SpinLock> guard(_lock);
		
		std::cerr << "Error: ";
		emitReasonParts(reasonParts...);
		std::cerr << std::endl;
		
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
		
		std::lock_guard<SpinLock> guard(_lock);
		
		std::cerr << "Error: ";
		emitReasonParts(reasonParts...);
		std::cerr << std::endl;
		
#ifndef NDEBUG
		abort();
#else
		exit(1);
#endif
	}
	
};


#endif // FATAL_ERROR_HANDLER_HPP
