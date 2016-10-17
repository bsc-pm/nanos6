#ifndef CPU_DEPENDENCY_DATA_HPP
#define CPU_DEPENDENCY_DATA_HPP


#include <atomic>
#include <bitset>
#include <deque>

#include "DataAccessRange.hpp"


class Task;


struct CPUDependencyData {
	struct DelayedOperation {
		enum operation_bit_number_t {
			PROPAGATE_READ = 0,
			PROPAGATE_WRITE,
			PROPAGATE_TO_FRAGMENTS,
			LINK_BOTTOM_ACCESSES_TO_NEXT,
			OPERATION_BIT_COUNT
		};
		
		std::bitset<OPERATION_BIT_COUNT> _operation;
		DataAccessRange _range;
		Task *_target;
		Task *_next;
		
		DelayedOperation()
			: _operation(), _range(), _target(nullptr), _next(nullptr)
		{
		}
	};
	
	
	typedef std::deque<DelayedOperation> delayed_operations_t;
	typedef std::deque<Task *> satisfied_originator_list_t;
	typedef std::deque<Task *> removable_task_list_t;
	
	//! Tasks whose accesses have been satified after ending a task
	satisfied_originator_list_t _satisfiedOriginators;
	delayed_operations_t _delayedOperations;
	removable_task_list_t _removableTasks;
	
#ifndef NDEBUG
	std::atomic<bool> _inUse;
#endif
};


#endif // CPU_DEPENDENCY_DATA_HPP
