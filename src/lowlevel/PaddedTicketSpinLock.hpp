/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef PADDED_TICKET_SPIN_LOCK_HPP
#define PADDED_TICKET_SPIN_LOCK_HPP


#include <cstddef>

#include "TicketSpinLock.hpp"
#include "Padding.hpp"

template <typename TICKET_T = uint16_t, int PADDING = CACHELINE_SIZE>
class PaddedTicketSpinLock {
private:
	typedef TicketSpinLock<TICKET_T> inner_type_t;

	char _frontPadding[PADDING - sizeof(inner_type_t)];
	inner_type_t _lock;
	char _backPadding[PADDING - sizeof(inner_type_t)];


public:
	inline PaddedTicketSpinLock()
	{
	}

	inline void lock()
	{
		_lock.lock();
	}

	inline bool tryLock()
	{
		return _lock.tryLock();
	}

	inline void unlock()
	{
		_lock.unlock();
	}

	inline inner_type_t &getTicketLock()
	{
		return _lock;
	}
};


#endif // PADDED_TICKET_SPIN_LOCK_HPP
