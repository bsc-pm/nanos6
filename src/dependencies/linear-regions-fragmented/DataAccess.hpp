#ifndef DATA_ACCESS_HPP
#define DATA_ACCESS_HPP


#include <boost/intrusive/avl_set.hpp>
#include <boost/intrusive/avl_set_hook.hpp>

#include <atomic>
#include <bitset>
#include <cassert>
#include <set>

#include <InstrumentDataAccessId.hpp>
#include <InstrumentTaskId.hpp>

#include <lowlevel/SpinLock.hpp>

struct DataAccess;
class Task;


#include "../DataAccessBase.hpp"
#include "DataAccessRange.hpp"


//! The accesses that one or more tasks perform sequentially to a memory location that can occur concurrently (unless commutative).
struct DataAccess : public DataAccessBase {
	enum status_bit_coding {
		COMPLETE_BIT = 0,
		READ_SATISFIED_BIT,
		WRITE_SATISFIED_BIT,
		FRAGMENT_BIT,
		HAS_SUBACCESSES_BIT,
		IN_BOTTOM_MAP,
		TOPMOST_BIT,
		FORCE_REMOVAL_BIT,
#ifndef NDEBUG
		IS_REACHABLE_BIT,
		HAS_BEEN_DISCOUNTED_BIT,
#endif
		TOTAL_STATUS_BITS
	};
	
	//! The range of data covered by the access
	DataAccessRange _range;
	
	typedef std::bitset<TOTAL_STATUS_BITS> status_t;
	status_t _status;
	
	//! Direct next access
	Task *_next;

	DataAccess(
		DataAccessType type, bool weak,
		Task *originator,
		DataAccessRange accessRange,
		bool fragment,
		Instrument::data_access_id_t instrumentationId
	)
		: DataAccessBase(type, weak, originator, instrumentationId),
		_range(accessRange),
		_status(0),
		_next(nullptr)
	{
		assert(originator != 0);
		
		if (fragment) {
			_status[FRAGMENT_BIT] = true;
		}
	}
	
	
	typename status_t::reference complete()
	{
		return _status[COMPLETE_BIT];
	}
	bool complete() const
	{
		return _status[COMPLETE_BIT];
	}
	
	typename status_t::reference readSatisfied()
	{
		return _status[READ_SATISFIED_BIT];
	}
	bool readSatisfied() const
	{
		return _status[READ_SATISFIED_BIT];
	}
	
	typename status_t::reference writeSatisfied()
	{
		return _status[WRITE_SATISFIED_BIT];
	}
	bool writeSatisfied() const
	{
		return _status[WRITE_SATISFIED_BIT];
	}
	
	bool isFragment() const
	{
		return _status[FRAGMENT_BIT];
	}
	
	typename status_t::reference hasSubaccesses()
	{
		return _status[HAS_SUBACCESSES_BIT];
	}
	bool hasSubaccesses() const
	{
		return _status[HAS_SUBACCESSES_BIT];
	}
	
	typename status_t::reference isInBottomMap()
	{
		return _status[IN_BOTTOM_MAP];
	}
	bool isInBottomMap() const
	{
		return _status[IN_BOTTOM_MAP];
	}
	
	
	typename status_t::reference isTopmost()
	{
		return _status[TOPMOST_BIT];
	}
	bool isTopmost() const
	{
		return _status[TOPMOST_BIT];
	}
	
	typename status_t::reference hasForcedRemoval()
	{
		return _status[FORCE_REMOVAL_BIT];
	}
	bool hasForcedRemoval() const
	{
		return _status[FORCE_REMOVAL_BIT];
	}
	
#ifndef NDEBUG
	typename status_t::reference isReachable()
	{
		return _status[IS_REACHABLE_BIT];
	}
	bool isReachable() const
	{
		return _status[IS_REACHABLE_BIT];
	}
#endif
	
	void markAsDiscounted()
	{
#ifndef NDEBUG
		assert(!_status[HAS_BEEN_DISCOUNTED_BIT]);
		_status[HAS_BEEN_DISCOUNTED_BIT] = true;
#endif
	}
	
#ifndef NDEBUG
	bool hasBeenDiscounted() const
	{
		return _status[HAS_BEEN_DISCOUNTED_BIT];
	}
#endif
	
	DataAccessRange const &getAccessRange() const
	{
		return _range;
	}
	
	DataAccessRange &getAccessRange()
	{
		return _range;
	}
	
	
	bool satisfied() const
	{
		if (_type == READ_ACCESS_TYPE) {
			return readSatisfied();
		} else {
			return readSatisfied() && writeSatisfied();
		}
	}
	
	
	bool isRemovable(bool hasForcedRemoval) const
	{
		return isTopmost() 
			&& readSatisfied()
			&& writeSatisfied()
			&& complete()
			&& ( !isInBottomMap() || hasForcedRemoval || (_next != nullptr) );
	}

};


#endif // DATA_ACCESS_HPP
