/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_DEPENDENCY_DATA_HPP
#define CPU_DEPENDENCY_DATA_HPP


#include <atomic>
#include <bitset>
#include <deque>
#include <boost/dynamic_bitset.hpp>

#include <limits.h>

#include "DataAccessLink.hpp"
#include "DataAccessRegion.hpp"


class Task;
class ReductionInfo;


struct CPUDependencyData {
	struct UpdateOperation {
		DataAccessLink _target;
		DataAccessRegion _region;
		
		bool _makeReadSatisfied;
		bool _makeWriteSatisfied;
		bool _makeConcurrentSatisfied;
		
		bool _makeTopmost;
		bool _makeTopLevel;
		
		bool _setReductionInfo; // Note: Both this and next field are required, as a null ReductionInfo can be propagated
		ReductionInfo *_reductionInfo;
		
		boost::dynamic_bitset<> _reductionCpuSet;
		
		UpdateOperation()
			: _target(), _region(),
			_makeReadSatisfied(false), _makeWriteSatisfied(false), _makeConcurrentSatisfied(false),
			_makeTopmost(false), _makeTopLevel(false),
			_setReductionInfo(false), _reductionInfo(nullptr)
		{
		}
		
		UpdateOperation(DataAccessLink const &target, DataAccessRegion const &region)
			: _target(target), _region(region),
			_makeReadSatisfied(false), _makeWriteSatisfied(false), _makeConcurrentSatisfied(false),
			_makeTopmost(false), _makeTopLevel(false),
			_setReductionInfo(false), _reductionInfo(nullptr)
		{
		}
		
		bool empty() const
		{
			return !_makeReadSatisfied && !_makeWriteSatisfied && !_makeConcurrentSatisfied
				&& !_makeTopmost && !_makeTopLevel
				&& !_setReductionInfo
				&& (_reductionCpuSet.size() == 0);
		}
	};
	
	
	typedef std::deque<UpdateOperation> delayed_operations_t;
	typedef std::deque<Task *> satisfied_originator_list_t;
	typedef std::deque<Task *> removable_task_list_t;
	
	//! Tasks whose accesses have been satisfied after ending a task
	satisfied_originator_list_t _satisfiedOriginators;
	delayed_operations_t _delayedOperations;
	removable_task_list_t _removableTasks;
	
#ifndef NDEBUG
	std::atomic<bool> _inUse;
#endif
	
	CPUDependencyData()
		: _satisfiedOriginators(), _delayedOperations(), _removableTasks()
#ifndef NDEBUG
		, _inUse(false)
#endif
	{
	}
	
	~CPUDependencyData()
	{
		assert(empty());
	}
	
	inline bool empty() const
	{
		return _satisfiedOriginators.empty() && _delayedOperations.empty() && _removableTasks.empty();
	}
};


#endif // CPU_DEPENDENCY_DATA_HPP
