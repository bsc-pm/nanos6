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
private:
	enum begin_tag_t {
		begin_tag_value
	};
	
	enum special_address_values_t : size_t {
		lowest_valid_address = 1024
	};
	
public:
	typedef void *address_t;
	
	#warning Backtraces are not supported in this platform
	
	enum {
		max_backtrace_frames = 0
	};
	
	enum {
		involves_libc_malloc = 0
	};
	
	class const_iterator {
	public:
		const_iterator()
		{
		}
		
		const_iterator(begin_tag_t)
		{
		}
		
		const_iterator &operator++()
		{
			return *this;
		}
		
		inline void *operator*() const
		{
			return nullptr;
		}
		
		inline bool operator==(const_iterator const &other)
		{
			// Assume always the end
			return true;
		}
		
		inline bool operator!=(const_iterator const &other)
		{
			// Assume always the end
			return false;
		}
	};
	
	
	static inline const_iterator begin()
	{
		return const_iterator(begin_tag_value);
	}
	
	static inline const_iterator end()
	{
		return const_iterator();
	}
};

} // namespace Instrument


#undef inline


#endif // INSTRUMENT_SUPPORT_BACKTRACE_UNSUPPORTED_BACKTRACE_WALKER_HPP
