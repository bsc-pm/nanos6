/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_REGISTRATION_HPP
#define DATA_ACCESS_REGISTRATION_HPP

#include "../DataAccessType.hpp"

class ComputePlace;
class Task;


namespace DataAccessRegistration {
	//! \brief creates a task data access taking into account repeated accesses but does not link it to previous accesses nor superaccesses
	//! 
	//! \param[in,out] task the task that performs the access
	//! \param[in] accessType the type of access
	//! \param[in] address the starting address of the access
	void registerTaskDataAccess(Task *task, DataAccessType accessType, void *address);
	
	//! \brief Performs the task dependency registration procedure
	//! 
	//! \param[in] task the Task whose dependencies need to be calculated
	//! 
	//! \returns true if the task is already ready
	bool registerTaskDataAccesses(Task *task, ComputePlace *computePlace);
	
	void unregisterTaskDataAccesses(Task *task, ComputePlace *computePlace);
	
	void handleEnterBlocking(Task *task);
	void handleExitBlocking(Task *task);
	void handleEnterTaskwait(Task *task, ComputePlace *computePlace);
	void handleExitTaskwait(Task *task, ComputePlace *computePlace);
	void handleTaskRemoval(Task *task, ComputePlace *computePlace);
}


#endif // DATA_ACCESS_REGISTRATION_HPP
