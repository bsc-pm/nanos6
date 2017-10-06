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
	struct PropagationBits {
		bool _read;
		bool _write;
		bool _concurrent;
		reduction_type_and_operator_index_t _reductionTypeAndOperatorIndex;
		bool _becomesRemovable;
		bool _makesNextTopmost;
		
		PropagationBits()
			: _read(false), _write(false), _concurrent(false), _reductionTypeAndOperatorIndex(no_reduction_type_and_operator),
			_becomesRemovable(false), _makesNextTopmost(false)
		{
		}
		
		inline bool propagates() const
		{
			return (_read || _write || _concurrent || (_reductionTypeAndOperatorIndex != no_reduction_type_and_operator) || _makesNextTopmost);
		}
		
		inline bool propagatesSatisfiability() const
		{
			return (_read || _write || _concurrent || (_reductionTypeAndOperatorIndex != no_reduction_type_and_operator));
		}
	};
	
	
	struct DelayedOperation {
		enum operation_type_t {
			link_bottom_map_accesses_operation,
			propagate_satisfiability_plain_operation,
			propagate_satisfiability_to_fragments_operation
		};
		
		operation_type_t _operationType;
		PropagationBits _propagationBits;
		
		Task *_next; // This is only for link_bottom_map_accesses_operation
		
		DataAccessRegion _region;
		Task *_target;
		
		DelayedOperation()
			: _propagationBits(),
			_next(nullptr),
			_region(), _target(nullptr)
		{
		}
	};
	
	
	typedef std::deque<DelayedOperation> delayed_operations_t;
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
};


#endif // CPU_DEPENDENCY_DATA_HPP
