/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_BACKTRACE_LIBUNWIND_BACKTRACE_WALKER_HPP
#define INSTRUMENT_SUPPORT_BACKTRACE_LIBUNWIND_BACKTRACE_WALKER_HPP


#define UNW_LOCAL_ONLY
#include <libunwind.h>


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
	
	enum {
		max_backtrace_frames = 1000
	};
	
	enum {
		involves_libc_malloc = 0
	};
	
	class const_iterator {
	private:
		unw_context_t _context;
		unw_cursor_t _cursor;
		unw_word_t _address;
		bool _valid;
		
	public:
		inline const_iterator()
			: _valid(false)
		{
		}
		
		inline const_iterator(begin_tag_t)
			: _valid(true)
		{
			unw_getcontext(&_context);
			unw_init_local(&_cursor, &_context);
		}
		
		inline const_iterator &operator++()
		{
			do {
				_valid = (unw_step(&_cursor) > 0);
				if (_valid) {
					_valid = (unw_get_reg(&_cursor, UNW_REG_IP, &_address) == 0);
				}
			} while (_valid && ((address_t) _address < (address_t) lowest_valid_address));
			
			return *this;
		}
		
		inline void *operator*() const
		{
			if (_valid) {
				return (void *) _address;
			} else {
				return nullptr;
			}
		}
		
		inline bool operator==(const_iterator const &other)
		{
			// WARNING: We only check whether we are at the end
			return (_valid == other._valid);
		}
		
		inline bool operator!=(const_iterator const &other)
		{
			// WARNING: We only check whether we are at the end
			return (_valid != other._valid);
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


#endif // INSTRUMENT_SUPPORT_BACKTRACE_LIBUNWIND_BACKTRACE_WALKER_HPP
