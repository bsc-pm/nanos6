#ifndef RW_TICKET_SPIN_LOCK_HPP
#define RW_TICKET_SPIN_LOCK_HPP



#include "TicketSpinLock.hpp"


class RWTicketSpinLock {
private:
	TicketSpinLock<> _readersSpinLock;
	long _readers;
	TicketSpinLock<> _writerSpinLock;
	
public:
	RWTicketSpinLock()
		: _readersSpinLock(), _readers(0), _writerSpinLock()
		{
		}
		
	inline void readLock()
	{
		_readersSpinLock.lock();
		if (++_readers == 1) {
			_writerSpinLock.lock();
		}
		_readersSpinLock.unlock();
	}
	
	inline void readUnlock()
	{
		_readersSpinLock.lock();
		if (--_readers == 0) {
			_writerSpinLock.unlock();
		}
		_readersSpinLock.unlock();
	}
	
	inline void writeLock()
	{
		_writerSpinLock.lock();
	}
	
	inline void writeUnlock()
	{
		_writerSpinLock.unlock();
	}
	
};


#endif // RW_SPIN_LOCK_HPP
