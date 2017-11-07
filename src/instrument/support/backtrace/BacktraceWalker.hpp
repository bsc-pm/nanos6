/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_BACKTRACE_BACKTRACE_WALKER_HPP
#define INSTRUMENT_SUPPORT_BACKTRACE_BACKTRACE_WALKER_HPP


#if HAVE_CONFIG_H
#include "config.h"
#endif


#if HAVE_EXECINFO_H
#include <execinfo.h>
#endif

#if HAVE_LIBUNWIND_H
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#endif

#include <cassert>


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
	
#if HAVE_LIBUNWIND_H
	enum {
		max_backtrace_frames = 1000
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
#elif defined(HAVE_EXECINFO_H) && defined(HAVE_BACKTRACE)
	enum {
		max_backtrace_frames = 20
		internal_max_backtrace_frames = max_backtrace_frames + 2
	};
	
	class const_iterator {
	private:
		address_t _buffer[internal_max_backtrace_frames];
		int _index;
		int _total;
		
	public:
		inline const_iterator()
			: _index(-1)
		{
		}
		
		inline const_iterator(begin_tag_t)
		{
			_total = backtrace(_buffer, internal_max_backtrace_frames);
			_index = 0;
		}
		
		inline const_iterator &operator++()
		{
			do {
				_index++;
				if (_index == _total) {
					_index = -1;
				}
			} while ((_index != -1) && (_buffer[_index] < (address_t) lowest_valid_address));
			
			return *this;
		}
		
		inline void *operator*() const
		{
			if (_index != -1) {
				return _buffer[_index];
			} else {
				return nullptr;
			}
		}
		
		inline bool operator==(const_iterator const &other)
		{
			// WARNING: We only check whether we are at the end
			return (_index == other._index);
		}
		
		inline bool operator!=(const_iterator const &other)
		{
			// WARNING: We only check whether we are at the end
			return (_index != other._index);
		}
	};
#else
	#warning Backtraces are not supported in this platform
	
	enum {
		max_backtrace_frames = 0
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
#endif
	
	
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


#endif // INSTRUMENT_SUPPORT_BACKTRACE_BACKTRACE_WALKER_HPP
