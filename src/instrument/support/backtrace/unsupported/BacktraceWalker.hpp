/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_BACKTRACE_UNSUPPORTED_BACKTRACE_WALKER_HPP
#define INSTRUMENT_SUPPORT_BACKTRACE_UNSUPPORTED_BACKTRACE_WALKER_HPP


// Attempt to avoid this code appearing in the backtrace
#define inline __attribute__((always_inline))


namespace Instrument {

class BacktraceWalker {
public:
	typedef void *address_t;
	
	enum {
		max_backtrace_frames = 0
	};
	
	enum {
		involves_libc_malloc = 0
	};
	
	template <typename CONSUMER_T>
	static inline void walk(
		__attribute__((unused)) int maxFrames,
		__attribute__((unused)) int skipFrames,
		__attribute__((unused)) CONSUMER_T consumer
	) {
		#warning Backtraces are not supported in this platform
	}
};

} // namespace Instrument


#undef inline


#endif // INSTRUMENT_SUPPORT_BACKTRACE_UNSUPPORTED_BACKTRACE_WALKER_HPP
