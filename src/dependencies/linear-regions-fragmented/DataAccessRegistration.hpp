/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef DATA_ACCESS_REGISTRATION_HPP
#define DATA_ACCESS_REGISTRATION_HPP

#include <DataAccessRegion.hpp>

#include "../DataAccessType.hpp"
#include "ReductionSpecific.hpp"
#include "CPUDependencyData.hpp"

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
		Task *task,
		DataAccessType accessType,
		bool weak,
		DataAccessRegion region,
		int symbolIndex,
		reduction_type_and_operator_index_t reductionTypeAndOperatorIndex,
		reduction_index_t reductionIndex);

	//! \brief Performs the task dependency registration procedure
	//!
	//! \param[in] task the Task whose dependencies need to be calculated
	//!
	//! \returns true if the task is already ready
	bool registerTaskDataAccesses(
		Task *task,
		ComputePlace *computePlace,
		CPUDependencyData &dependencyData);

	void releaseAccessRegion(
		Task *task, DataAccessRegion region,
		DataAccessType accessType,
		bool weak,
		ComputePlace *computePlace,
		CPUDependencyData &dependencyData,
		MemoryPlace const *location = nullptr);

	void unregisterTaskDataAccesses(
		Task *task,
		ComputePlace *computePlace,
		CPUDependencyData &dependencyData,
		MemoryPlace *location = nullptr,
		bool fromBusyThread = false);

	//! \brief Combines the task reductions without releasing the dependencies
	//!
	//! \param[in] task the Task whose reductions need to be combined
	//! \param[in] computePlace the ComputePlace assigned to the current thread and where the task has been executed
	void combineTaskReductions(Task *task, ComputePlace *computePlace);

	//! \brief propagates satisfiability for an access.
	//!
	//! \param[in] task is the Task that includes the access for which we propagate.
	//! \param[in] region is the region for which propagate satisfiability
	//! \param[in] computePlace is the ComputePlace assigned to the current thread, or nullptr if none assigned
	//! \param[in] dependencyData is the CPUDependencyData struct used for the propagation operation.
	//! \param[in] readSatisfied is true if the region becomes read satisfied.
	//! \param[in] writeSatisfied is true if the region becomes write satisfied.
	//! \param[in] location is not a nullptr if we have an update for the location of the region.
	void propagateSatisfiability(
		Task *task,
		DataAccessRegion const &region,
		ComputePlace *computePlace,
		CPUDependencyData &dependencyData,
		bool readSatisfied,
		bool writeSatisfied,
		MemoryPlace const *location);

	void handleEnterTaskwait(Task *task, ComputePlace *computePlace, CPUDependencyData &dependencyData);
	void handleExitTaskwait(Task *task, ComputePlace *computePlace, CPUDependencyData &dependencyData);

	//! \brief Mark a Taskwait fragment as completed
	//!
	//! \param[in] task is the Task that created the taskwait fragment
	//! \param[in] region is the taskwait region that has been completed
	//! \param[in] computePlace is the current ComputePlace of the caller
	//! \param[in] hpDependencyData is the CPUDependencyData used for delayed operations
	void releaseTaskwaitFragment(
		Task *task,
		DataAccessRegion region,
		ComputePlace *computePlace,
		CPUDependencyData &hpDependencyData);

	//! \brief Pass all data accesses from the task through a lambda
	//!
	//! \param[in] task the owner of the accesses to be processed
	//! \param[in] processor a lambda that receives a reference to the access
	//!
	//! \returns false if the traversal was stopped before finishing
	template <typename ProcessorType>
	inline bool processAllDataAccesses(Task *task, ProcessorType processor);

	//! \brief Update the location of the DataAccess of a Task
	//!
	//! \param[in] task is the owner of the accesses we are updating
	//! \param[in] region is the DataAccessRegion of which the location we are updating
	//! \param[in] location is the new location of the DataAccess
	//! \param[in] isTaskwait is true if the update refers to a taskwait object
	void updateTaskDataAccessLocation(Task *task,
		DataAccessRegion const &region,
		MemoryPlace const *location,
		bool isTaskwait);

	//! \brief Register a region as a NO_ACCESS_TYPE access within the Task
	//!
	//! This is meant to be used for registering a new DataAccess that
	//! represents a new memory allocation in the context of a task. This
	//! way we can keep track of the location of that region in situations
	//! that we loose all information about it, e.g. after a taskwait
	//!
	//! \param[in] task is the Task that registers the access region
	//! \param[in] region is the DataAccessRegion being registered
	void registerLocalAccess(Task *task, DataAccessRegion const &region);

	//! \brief Unregister a local region from the accesses of the Task
	//!
	//! This is meant to be used for unregistering a DataAccess with
	//! NO_ACCESS_TYPE that was previously registered calling
	//! 'registerLocalAccess', when we are done with this region, i.e. the
	//! corresponging memory has been deallocated.
	//!
	//! \param[in] task is the Task that region is registerd to
	//! \param[in] region is the DataAccessRegion being unregistered
	void unregisterLocalAccess(Task *task, DataAccessRegion const &region);
} // namespace DataAccessRegistration


#endif // DATA_ACCESS_REGISTRATION_HPP
