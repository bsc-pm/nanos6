/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_TASK_EXECUTION_HPP
#define INSTRUMENT_TASK_EXECUTION_HPP


#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>

#include "InstrumentComputePlaceId.hpp"


namespace Instrument {

	//! This function is called just before start executing a task
	//! Task Hardware Counters are always updated before calling this function
	//! \param[in] taskid The task identifier
	void startTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called just after a task has finished
	//! Runtime Hardware Counters are always updated before calling this function
	//! \param[in] taskid The task identifier
	void endTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called when tasks resources are no longer needed
	//! \param[in] taskid The task identifier
	void destroyTask(task_id_t taskId, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called just before a collaborator starts executing a chunk
	//! Task Hardware Counters are always updated before calling this function
	//! \param[in] taskforId The task identifier
	//! \param[in] collaboratorId The collaborator identifier
	//! \param[in] first Whether the chunk is the first one
	void startTaskforCollaborator(task_id_t taskforId, task_id_t collaboratorId, bool first = false, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());

	//! This function is called just before a collaborator finishes it's chunk
	//! Runtime Hardware Counters are always updated before calling this function
	//! \param[in] taskforId The task identifier
	//! \param[in] collaboratorId The collaborator identifier
	//! \param[in] last Whether the chunk is the last one
	void endTaskforCollaborator(task_id_t taskforId, task_id_t collaboratorId, bool last = false, InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent());
}


#endif // INSTRUMENT_TASK_EXECUTION_HPP
