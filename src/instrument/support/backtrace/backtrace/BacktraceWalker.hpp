/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_BACKTRACE_BACKTRACE_BACKTRACE_WALKER_HPP
#define INSTRUMENT_SUPPORT_BACKTRACE_BACKTRACE_BACKTRACE_WALKER_HPP


#include <execinfo.h>

#include <climits>
#include <cstddef>


// Attempt to avoid this code appearing in the backtrace
#define inline __attribute__((always_inline))


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
		involves_libc_malloc = 1
	};
	
	
	template <typename CONSUMER_T>
	static inline void walk(int maxFrames, int skipFrames, CONSUMER_T consumer)
	{
		int allocatedFrames = (maxFrames + skipFrames) * 2;
		address_t buffer[allocatedFrames];
		
		int total = backtrace(buffer, allocatedFrames);
		int currentIndex = 0;
		int currentFrame = 0;
		
		// Skip as many frames as necessary
		while ((currentFrame < skipFrames) && (currentIndex < allocatedFrames) && (currentIndex < total)) {
			if (buffer[currentIndex] >= (address_t) lowest_valid_address) {
				currentFrame++;
			}
			currentIndex++;
		}
		
		// Process
		currentFrame = 0;
		while ((currentFrame < maxFrames) && (currentIndex < allocatedFrames) && (currentIndex < total)) {
			if (buffer[currentIndex] >= (address_t) lowest_valid_address) {
				consumer(buffer[currentIndex], currentFrame);
				currentFrame++;
			}
			currentIndex++;
		}
	}
};

} // namespace Instrument


#undef inline


#endif // INSTRUMENT_SUPPORT_BACKTRACE_BACKTRACE_BACKTRACE_WALKER_HPP
