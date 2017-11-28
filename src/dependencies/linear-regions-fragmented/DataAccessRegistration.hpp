/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_REGISTRATION_HPP
#define DATA_ACCESS_REGISTRATION_HPP

#include <DataAccessRegion.hpp>

#include "../DataAccessType.hpp"
#include "ReductionSpecific.hpp"

class ComputePlace;
class Task;


namespace DataAccessRegistration {
	//! \brief creates a task data access taking into account repeated accesses but does not link it to previous accesses nor superaccesses
	//! 
	//! \param[in,out] task the task that performs the access
	//! \param[in] accessType the type of access
	//! \param[in] weak true iff the access is weak
	//! \param[in] region the region of data covered by the access
	//! \param[in] reductionTypeAndOperatorIndex an index that identifies the type and the operation of the reduction
	//! \param[in] reductionIndex an index that identifies the reduction within the task
	void registerTaskDataAccess(
		Task *task, DataAccessType accessType, bool weak, DataAccessRegion region, int symbolIndex, reduction_type_and_operator_index_t reductionTypeAndOperatorIndex, reduction_index_t reductionIndex
	);
	
	//! \brief Performs the task dependency registration procedure
	//! 
	//! \param[in] task the Task whose dependencies need to be calculated
	//! 
	//! \returns true if the task is already ready
	bool registerTaskDataAccesses(
		Task *task,
		ComputePlace *computePlace
	);
	
	void releaseAccessRegion(
		Task *task, DataAccessRegion region,
		__attribute__((unused)) DataAccessType accessType, __attribute__((unused)) bool weak,
		ComputePlace *computePlace
	);
	
	void unregisterTaskDataAccesses(Task *task, ComputePlace *computePlace);
	
	void handleEnterBlocking(Task *task);
	void handleExitBlocking(Task *task);
	void handleEnterTaskwait(Task *task, ComputePlace *computePlace);
	void handleExitTaskwait(Task *task, __attribute__((unused)) ComputePlace *computePlace);
	
	static inline void handleTaskRemoval(
			__attribute__((unused)) Task *task,
			__attribute__((unused)) ComputePlace *computePlace
	) {
	}	
}


#endif // DATA_ACCESS_REGISTRATION_HPP
