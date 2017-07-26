/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_DEPENDENCIES_BY_GROUP_HPP
#define INSTRUMENT_DEPENDENCIES_BY_GROUP_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>


namespace Instrument {
	//! \file
	//! \name Dependency instrumentation by grouped accesses
	//! @{
	//! This group of functions is useful for instrumenting task dependencies. The main concepts are the group of tasks,
	//! and the sequence of groups of tasks. Such a sequence is identified by a parentTaskId and a handler. The parentTaskId
	//! determines the domain in which the dependencies are calculated, and the handler identifies the storage on that domain.
	//! 
	//! A group of tasks is determined by the tasks added in calls to Instrument::addTaskToAccessGroup in between two calls to
	//! Instrument::beginAccessGroup with the same handler.
	//! 
	//! A task may be contained in more than one group of tasks. These groups are used for determining the dependencies.
	//! The tasks in a group of a sequence are successors to the tasks of the previous group in the sequence and
	//! predecessors to the following one.
	//!
	//! This approach to instrumenting dependencies is not exhaustive, since the detection of groups depends on the
	//! availability of at least one unfinished task that accesses the storage. When this is not possible,
	//! Instrument::beginAccessGroup is called with sequenceIsEmpty set to true to indicate that the new group may not truly
	//! correspond to a real group but may be instead an artifact caused by the lack of tasks accessing the storage that
	//! is associated to the sequence.
	//! 
	//! This problem can be mitigated by also instrumenting the taskwaits.
	
	//! \brief Called when an address is about to have a tasks added that depend on all previously added ones since the last call to this function with the same handler
	//! 
	//! The parentTaskId parameter is useful in combination with the instrumentation of taskwaits to detect when
	//! a group with sequenceIsEmty set to true is due to the presece of a taswait, and therefore not to the work
	//! being depleted.
	//! 
	//! \param[in] parentTaskId the identifier of the parent task, that is the one that determines the domain of dependencies
	//! \param[in] handler an identifier for the sequence of groups of task accesses
	//! \param[in] sequenceIsEmpty true if the sequence of accesses is empty and that has caused a new group (which may be artificial)
	void beginAccessGroup(task_id_t parentTaskId, void *handler, bool sequenceIsEmpty, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	//! \brief Register a task in the current group of tasks associated with the given handler
	//! 
	//! \param[in] handler the identifier for the sequence of groups of task accesses
	//! \param[in] taskId the identifier of a task (returned by the call to Instrument::enterAddTask) that participates in the current group of accesses associated with the handler
	void addTaskToAccessGroup(void *handler, task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	//! \brief Unregister a task in the current group of tasks associated with the given handler
	//! 
	//! The task to be removed is the last one. The reason is that its access belongs to the next group that is about to be created.
	//! 
	//! \param[in] handler the identifier for the sequence of groups of task accesses
	//! \param[in] taskId the identifier of a task (returned by the call to Instrument::enterAddTask) that no longer participates in the current group of accesses associated with the handler
	void removeTaskFromAccessGroup(void *handler, task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
	
	//! @}
}


#endif // INSTRUMENT_DEPENDENCIES_BY_GROUP_HPP
