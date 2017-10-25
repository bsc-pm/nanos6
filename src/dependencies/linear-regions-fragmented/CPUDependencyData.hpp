/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef CPU_DEPENDENCY_DATA_HPP
#define CPU_DEPENDENCY_DATA_HPP


#include <atomic>
#include <bitset>
#include <deque>

#include <limits.h>

#include "DataAccessRegion.hpp"
#include "ReductionSpecific.hpp"


class Task;


struct CPUDependencyData {
	struct UpdateOperation {
		Task *_task;
		DataAccessRegion _region;
		
		bool _toAccesses; // As opposed to "to fragments"
		
		bool _makeReadSatisfied;
		bool _makeWriteSatisfied;
		bool _makeConcurrentSatisfied;
		
		bool _makeTopmost;
		
		reduction_type_and_operator_index_t _makeReductionSatisfied;
		
		UpdateOperation()
			: _task(nullptr), _region(), _toAccesses(true),
			_makeReadSatisfied(false), _makeWriteSatisfied(false), _makeConcurrentSatisfied(false),
			_makeTopmost(false),
			_makeReductionSatisfied(no_reduction_type_and_operator)
		{
		}
		
		UpdateOperation(Task *task, DataAccessRegion const &region, bool toAccesses)
			: _task(task), _region(region), _toAccesses(toAccesses),
			_makeReadSatisfied(false), _makeWriteSatisfied(false), _makeConcurrentSatisfied(false),
			_makeTopmost(false),
			_makeReductionSatisfied(no_reduction_type_and_operator)
		{
		}
		
		bool empty() const
		{
			return !_makeReadSatisfied && !_makeWriteSatisfied && !_makeConcurrentSatisfied
				&& !_makeTopmost
				&& (_makeReductionSatisfied == no_reduction_type_and_operator);
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
	
	inline bool empty() const
	{
		return _satisfiedOriginators.empty() && _delayedOperations.empty() && _removableTasks.empty();
	}
};


#endif // CPU_DEPENDENCY_DATA_HPP
