/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_BACKTRACE_LIBUNWIND_BACKTRACE_WALKER_HPP
#define INSTRUMENT_SUPPORT_BACKTRACE_LIBUNWIND_BACKTRACE_WALKER_HPP


#define UNW_LOCAL_ONLY
#include <libunwind.h>

#include <climits>


namespace Instrument {


class BacktraceWalker {
private:
	enum special_address_values_t : size_t {
		lowest_valid_address = 1024
	};
	
public:
	typedef void *address_t;
	
	enum {
		max_backtrace_frames = INT_MAX
	};
	
	enum {
		involves_libc_malloc = 0
	};
	
	template <typename CONSUMER_T>
	__attribute__((noinline)) static void walk(int maxFrames, int skipFrames, CONSUMER_T consumer)
	{
		// This method is not inline because some implementations of unw_getcontext are expanded to a call
		// to getcontext, which cause GCC to fail to inline the contaning function.
		skipFrames++;
		
		unw_context_t _context;
		unw_cursor_t _cursor;
		unw_word_t address;
		
		unw_getcontext(&_context);
		unw_init_local(&_cursor, &_context);
		
		int currentFrame = 0;
		
		bool valid = true;
		while (valid && (currentFrame < skipFrames)) {
			valid = (unw_step(&_cursor) > 0);
			currentFrame++;
		}
		
		currentFrame = 0;
		while (valid && (currentFrame < maxFrames)) {
			do {
				valid = (unw_get_reg(&_cursor, UNW_REG_IP, &address) == 0);
				if ((address_t) address < (address_t) lowest_valid_address) {
					valid = (unw_step(&_cursor) > 0);
				}
			} while (valid && ((address_t) address < (address_t) lowest_valid_address));
			
			if (valid) {
				consumer((address_t) address, currentFrame);
				valid = (unw_step(&_cursor) > 0);
				currentFrame++;
			}
		}
	}
};

} // namespace Instrument


#undef inline


#endif // INSTRUMENT_SUPPORT_BACKTRACE_LIBUNWIND_BACKTRACE_WALKER_HPP
