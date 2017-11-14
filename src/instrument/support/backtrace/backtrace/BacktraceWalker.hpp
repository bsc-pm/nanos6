/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_SUPPORT_BACKTRACE_BACKTRACE_BACKTRACE_WALKER_HPP
#define INSTRUMENT_SUPPORT_BACKTRACE_BACKTRACE_BACKTRACE_WALKER_HPP


#include <execinfo.h>

#include <cstddef>


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
		max_backtrace_frames = 20,
		internal_max_backtrace_frames = max_backtrace_frames + 2
	};
	
	enum {
		involves_libc_malloc = 1
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


#endif // INSTRUMENT_SUPPORT_BACKTRACE_BACKTRACE_BACKTRACE_WALKER_HPP
