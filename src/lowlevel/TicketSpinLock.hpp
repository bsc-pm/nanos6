/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef TICKET_SPIN_LOCK_HPP
#define TICKET_SPIN_LOCK_HPP


#include <atomic>
#include <cassert>
#include <cstddef>


#ifndef SPIN_LOCK_READS_BETWEEN_CMPXCHG
#define SPIN_LOCK_READS_BETWEEN_CMPXCHG 1000
#endif


#ifndef NDEBUG
class WorkerThread;
namespace ompss_debug {
	__attribute__((weak)) WorkerThread *getCurrentWorkerThread();
}
#endif


template <typename TICKET_T = uint16_t>
class TicketSpinLock {
private:
	std::atomic<TICKET_T> _currentTicket;
	std::atomic<TICKET_T> _nextFreeTicket;
	
#ifndef NDEBUG
	WorkerThread *_owner;
#endif
	
	inline void assertCurrentOwner();
	inline void assertUnowned();
	inline void assertUnownedOrCurrentOwner();
	inline void setOwner();
	inline void unsetOwner();
	
public:
	TicketSpinLock()
		: _currentTicket(0), _nextFreeTicket(0)
#ifndef NDEBUG
		, _owner(nullptr)
#endif
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
			int spinsLeft = SPIN_LOCK_READS_BETWEEN_CMPXCHG;
			TICKET_T current;
			do {
				current = _currentTicket.load(std::memory_order_relaxed);
				spinsLeft--;
			} while ((current != ticket) && (spinsLeft > 0));
		}
		
		assertUnowned();
		setOwner();
	}
	
	inline bool tryLock()
	{
		TICKET_T ticket = _nextFreeTicket;
		
		if (_currentTicket.load() == ticket) {
			if (_nextFreeTicket.compare_exchange_strong(ticket, ticket+1)) {
				assertUnowned();
				setOwner();
				
				return true;
			} else {
				return false;
			}
		} else {
			return false;
		}
	}
	
	inline void unlock()
	{
		assertCurrentOwner();
		unsetOwner();
		
		_currentTicket.fetch_add(1, std::memory_order_release);
	}
	
	inline bool isLockedByThisThread();
};


#ifndef NDEBUG
template <typename TICKET_T>
inline void TicketSpinLock<TICKET_T>::assertCurrentOwner()
{
	assert(_owner == ompss_debug::getCurrentWorkerThread());
}

template <typename TICKET_T>
inline void TicketSpinLock<TICKET_T>::assertUnowned()
{
	assert(_owner == nullptr);
}

template <typename TICKET_T>
inline void TicketSpinLock<TICKET_T>::assertUnownedOrCurrentOwner()
{
	assert( (_owner == nullptr) || (_owner == ompss_debug::getCurrentWorkerThread()) ) ;
}

template <typename TICKET_T>
inline void TicketSpinLock<TICKET_T>::setOwner()
{
	_owner = ompss_debug::getCurrentWorkerThread();
}

template <typename TICKET_T>
inline void TicketSpinLock<TICKET_T>::unsetOwner()
{
	_owner = nullptr;
}
#else
template <typename TICKET_T>
inline void TicketSpinLock<TICKET_T>::assertCurrentOwner()
{
}

template <typename TICKET_T>
inline void TicketSpinLock<TICKET_T>::assertUnowned()
{
}

template <typename TICKET_T>
inline void TicketSpinLock<TICKET_T>::assertUnownedOrCurrentOwner()
{
}

template <typename TICKET_T>
inline void TicketSpinLock<TICKET_T>::setOwner()
{
}

template <typename TICKET_T>
inline void TicketSpinLock<TICKET_T>::unsetOwner()
{
}
#endif


#ifndef NDEBUG
template <typename TICKET_T>
inline bool TicketSpinLock<TICKET_T>::isLockedByThisThread()
{
	return (_owner == ompss_debug::getCurrentWorkerThread());
}
#endif


#endif // TICKET_SPIN_LOCK_HPP
