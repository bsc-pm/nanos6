/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_CTF_BLOCKING_API_HPP
#define INSTRUMENT_CTF_BLOCKING_API_HPP


#include "../api/InstrumentBlockingAPI.hpp"
#include "CTFTracepoints.hpp"


namespace Instrument {
	inline void enterBlockCurrentTask(
		__attribute__((unused)) task_id_t taskId,
		bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		if (taskRuntimeTransition)
			tp_blocking_api_block_tc_enter();
	}

	inline void exitBlockCurrentTask(
		__attribute__((unused)) task_id_t taskId,
		bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		if (taskRuntimeTransition)
			tp_blocking_api_block_tc_exit();
	}

	inline void enterUnblockTask(
		__attribute__((unused)) task_id_t taskId,
		bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		if (taskRuntimeTransition) {
			tp_blocking_api_unblock_tc_enter();
		} else {
			tp_blocking_api_unblock_oc_enter();
		}
	}

	inline void exitUnblockTask(
		__attribute__((unused)) task_id_t taskId,
		bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		if (taskRuntimeTransition) {
			tp_blocking_api_unblock_tc_exit();
		} else {
			tp_blocking_api_unblock_oc_exit();
		}
	}

	inline void enterWaitFor(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		tp_waitfor_tc_enter();
	}

	inline void exitWaitFor(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
		tp_waitfor_tc_exit();
	}
}


#endif // INSTRUMENT_CTF_BLOCKING_HPP
