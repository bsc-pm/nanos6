/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2018-2020 Barcelona Supercomputing Center (BSC)
*/

#include "InstrumentBlockingAPI.hpp"
#include "InstrumentVerbose.hpp"

#include <InstrumentInstrumentationContext.hpp>


using namespace Instrument::Verbose;



namespace Instrument {
	void enterBlockCurrentTask(
		task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		InstrumentationContext const &context
	) {
		if (!_verboseBlocking) {
			return;
		}

		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);

		logEntry->appendLocation(context);
		logEntry->_contents << " --> nanos6_block_current_task task:" << taskId;

		addLogEntry(logEntry);
	}

	void exitBlockCurrentTask(
		task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		InstrumentationContext const &context
	) {
		if (!_verboseBlocking) {
			return;
		}

		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);

		logEntry->appendLocation(context);
		logEntry->_contents << " <-- nanos6_block_current_task task:" << taskId;

		addLogEntry(logEntry);
	}

	void enterUnblockTask(
		task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		InstrumentationContext const &context
	) {
		if (!_verboseBlocking) {
			return;
		}

		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);

		logEntry->appendLocation(context);
		logEntry->_contents << " <-> nanos6_unblock_task task:" << taskId;

		addLogEntry(logEntry);
	}

	void exitUnblockTask(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) bool taskRuntimeTransition,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	void enterWaitFor(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}

	void exitWaitFor(
		__attribute__((unused)) task_id_t taskId,
		__attribute__((unused)) InstrumentationContext const &context
	) {
	}
}

