#ifndef CPU_DEPENDENCY_DATA_HPP
#define CPU_DEPENDENCY_DATA_HPP


#include <atomic>
#include <bitset>
#include <deque>

#include "DataAccessRange.hpp"


class Task;


struct CPUDependencyData {
	struct DelayedOperation {
		enum operation_type_t {
			link_bottom_map_accesses_operation,
			propagate_satisfiability_plain_operation,
			propagate_satisfiability_to_fragments_operation
		};
		
		operation_type_t _operationType;
		bool _propagateRead;
		bool _propagateWrite;
		bool _makeTopmost;
		Task *_next; // This is only for link_bottom_map_accesses_operation
		
		DataAccessRange _range;
		Task *_target;
		
		DelayedOperation()
			: _propagateRead(false), _propagateWrite(false), _makeTopmost(false),
			_next(nullptr),
			_range(), _target(nullptr)
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
