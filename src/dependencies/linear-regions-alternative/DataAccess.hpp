/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

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
#include "DataAccessRegion.hpp"


//! The accesses that one or more tasks perform sequentially to a memory location that can occur concurrently (unless commutative).
struct DataAccess : public DataAccessBase {
	enum status_bit_coding {
		COMPLETE_BIT = 0,
		READ_SATISFIED_BIT,
		WRITE_SATISFIED_BIT,
		TOPMOST_SATISFIED_BIT,
		HAS_SUBACCESSES_BIT,
		IN_BOTTOM_MAP,
#ifndef NDEBUG
		IS_REACHABLE_BIT,
		HAS_BEEN_DISCOUNTED_BIT,
#endif
		TOTAL_STATUS_BITS
	};
	
	//! The region of data covered by the access
	DataAccessRegion _region;
	
	typedef std::bitset<TOTAL_STATUS_BITS> status_t;
	status_t _status;
	
	//! Direct next access
	Task *_next;
	
	//! First child with accesses within this region
	Task *_child;
	
	//! Reduction type and operator
	int _reductionInfo;
	
	DataAccess(
		DataAccessType type, bool weak,
		Task *originator,
		DataAccessRegion accessRegion,
		Instrument::data_access_id_t instrumentationId
	)
		: DataAccessBase(type, weak, originator, instrumentationId),
		_region(accessRegion),
		_status(0),
		_next(nullptr),
		_child(nullptr),
		_reductionInfo(0)
	{
		assert(originator != 0);
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
	
	typename status_t::reference topmostSatisfied()
	{
		return _status[TOPMOST_SATISFIED_BIT];
	}
	bool topmostSatisfied() const
	{
		return _status[TOPMOST_SATISFIED_BIT];
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
	
	void setAccessRegion(DataAccessRegion const &newRegion)
	{
		_region = newRegion;
	}
	
	DataAccessRegion const &getAccessRegion() const
	{
		return _region;
	}
	
	DataAccessRegion &getAccessRegion()
	{
		return _region;
	}
	
	
	bool satisfied() const
	{
		if (_type == READ_ACCESS_TYPE) {
			return readSatisfied();
		} else {
			return readSatisfied() && writeSatisfied();
		}
	}
	
	
	bool isRemovable() const
	{
		return readSatisfied()
			&& writeSatisfied()
			&& topmostSatisfied()
			&& complete()
			&& !hasSubaccesses();
	}
	
};


#endif // DATA_ACCESS_HPP
