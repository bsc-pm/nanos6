#ifndef FIXED_ADDRESS_DATA_ACCESS_MAP_HPP
#define FIXED_ADDRESS_DATA_ACCESS_MAP_HPP


#include "DataAccessRange.hpp"
#include "DataAccessSequence.hpp"
#include "DiscreteAddressMap.hpp"
#include "lowlevel/SpinLock.hpp"


struct FixedAddressDataAccessMapNodeContents {
	SpinLock _lock;
	DataAccessSequence _accessSequence;
	
	FixedAddressDataAccessMapNodeContents()
		: _lock(), _accessSequence(&_lock)
	{
	}
	
	FixedAddressDataAccessMapNodeContents(DataAccessRange accessRange)
		: _lock(), _accessSequence(accessRange, &_lock)
	{
	}
	
	DataAccessRange const &getAccessRange() const
	{
		return _accessSequence._accessRange;
	}
};


typedef DiscreteAddressMap<FixedAddressDataAccessMapNodeContents> FixedAddressDataAccessMap;


#endif // FIXED_ADDRESS_DATA_ACCESS_MAP_HPP
