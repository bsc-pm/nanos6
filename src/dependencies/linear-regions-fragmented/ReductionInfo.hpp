/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef REDUCTION_INFO_HPP
#define REDUCTION_INFO_HPP

#include <vector>
#include <functional>
#include <atomic>

#include <DataAccessRegion.hpp>
#include <lowlevel/SpinLock.hpp>

#include <boost/dynamic_bitset.hpp>

#include "ReductionSpecific.hpp"

class ReductionInfo
{
	public:
		
		typedef SpinLock spinlock_t;
		
		ReductionInfo(DataAccessRegion region, reduction_type_and_operator_index_t typeAndOperatorIndex,
				std::function<void(void*, void*, size_t)> initializationFunction, std::function<void(void*, void*, size_t)> combinationFunction);
		
		~ReductionInfo();
		
		reduction_type_and_operator_index_t getTypeAndOperatorIndex() const;
		
		const DataAccessRegion& getOriginalRegion() const;
		
		DataAccessRegion getCPUPrivateStorage(size_t virtualCpuId);
		
		bool combineRegion(const DataAccessRegion& region, const boost::dynamic_bitset<>& reductionCpuSet);
		
	private:
		
		DataAccessRegion _region;
		
		DataAccessRegion _storage;
		
		boost::dynamic_bitset<> _isCpuStorageInitialized;
		
		std::atomic_size_t _sizeCounter;
		
		reduction_type_and_operator_index_t _typeAndOperatorIndex;
		
		std::function<void(void*, size_t)> _initializationFunction;
		
		std::function<void(void*, void*, size_t)> _combinationFunction;
		
		spinlock_t _lock;
};

#endif // REDUCTION_INFO_HPP
