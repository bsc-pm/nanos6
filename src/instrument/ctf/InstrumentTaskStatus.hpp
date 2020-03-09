/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.
	
	Copyright (C) 2015-2017 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_TASK_STATUS_HPP
#define INSTRUMENT_CTF_TASK_STATUS_HPP

#include <CTFAPI.hpp>
#include "../api/InstrumentTaskStatus.hpp"


namespace Instrument {
	inline void taskIsPending(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void taskIsReady(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void taskIsExecuting(
		task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		CTFTaskInfo *ctfTaskInfo = taskId._ctfTaskInfo;
		assert(CTFTaskInfo != nullptr);
		nanos6_task_info_t *nanos6TaskInfo = ctfTaskInfo->_nanos6TaskInfo;
		assert(nanos6TaskInfo != nullptr);
		CTFAPI::tracepoint(TP_NANOS6_TASK_EXECUTE, reinterpret_cast<uint64_t>(nanos6TaskInfo->implementations[0].run), static_cast<uint64_t>(ctfTaskInfo->_taskId));
	}
	
	inline void taskIsBlocked(
		task_id_t taskId,
		__attribute__((unused)) task_blocking_reason_t reason,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		CTFTaskInfo *ctfTaskInfo = taskId._ctfTaskInfo;
		assert(CTFTaskInfo != nullptr);
		CTFAPI::tracepoint(TP_NANOS6_TASK_BLOCK, static_cast<uint64_t>(ctfTaskInfo->_taskId));
	}
	
	inline void taskIsZombie(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void taskIsBeingDeleted(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void taskHasNewPriority(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) long priority,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void taskforCollaboratorIsExecuting(
		__attribute__((unused)) task_id_t taskforId,
		__attribute__((unused)) task_id_t collaboratorId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
	
	inline void taskforCollaboratorStopped(
		__attribute__((unused)) task_id_t taskforId,
		__attribute__((unused)) task_id_t collaboratorId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
}


#endif // INSTRUMENT_CTF_TASK_STATUS_HPP
