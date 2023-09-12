/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef TICKET_SPIN_LOCK_HPP
#define TICKET_SPIN_LOCK_HPP


#include <atomic>
#include <cassert>
#include <cstddef>

#include "SpinLock.hpp"


template <typename TICKET_T = uint16_t>
class TicketSpinLock {
private:
	std::atomic<TICKET_T> _currentTicket;
	std::atomic<TICKET_T> _nextFreeTicket;

public:
	TicketSpinLock() : _currentTicket(0), _nextFreeTicket(0)
	{
	}

	~TicketSpinLock()
	{
		// Not locked
		assert(_currentTicket == _nextFreeTicket);
	}

	inline void lock()
	{
		TICKET_T ticket = _nextFreeTicket++;

		while (_currentTicket.load(std::memory_order_acquire) != ticket) {
			int spinsLeft = SpinLock::ReadsBetweenCompareExchange;
			TICKET_T current;
			do {
				current = _currentTicket.load(std::memory_order_relaxed);
				spinsLeft--;
			} while ((current != ticket) && (spinsLeft > 0));
		}
	}

	inline bool tryLock()
	{
		TICKET_T ticket = _nextFreeTicket;

		if (_currentTicket.load() != ticket)
			return false;

		return _nextFreeTicket.compare_exchange_strong(ticket, ticket+1);
	}

	inline void unlock()
	{
		_currentTicket.fetch_add(1, std::memory_order_release);
	}
};

#endif // TICKET_SPIN_LOCK_HPP
