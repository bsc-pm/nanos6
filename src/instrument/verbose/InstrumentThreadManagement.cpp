/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#include <cassert>

#include <InstrumentInstrumentationContext.hpp>
#include <InstrumentThreadInstrumentationContext.hpp>

#include "InstrumentThreadManagement.hpp"
#include "InstrumentVerbose.hpp"
#include "executors/threads/CPU.hpp"
#include "executors/threads/WorkerThread.hpp"


using namespace Instrument::Verbose;


namespace Instrument {
	void enterThreadCreation(/* OUT */ thread_id_t &threadId, __attribute__((unused)) compute_place_id_t const &computePlaceId) {
		threadId = GenericIds::getNewThreadId();

		if (!_verboseThreadManagement) {
			return;
		}

		InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent();

		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);

		logEntry->appendLocation(context);
		logEntry->_contents << " --> CreateThread " << threadId;

		addLogEntry(logEntry);
	}


	void exitThreadCreation(thread_id_t threadId) {
		threadId = GenericIds::getNewThreadId();

		if (!_verboseThreadManagement) {
			return;
		}

		InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent();

		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);

		logEntry->appendLocation(context);
		logEntry->_contents << " <-- CreateThread " << threadId;

		addLogEntry(logEntry);
	}


	void createdThread(thread_id_t threadId, compute_place_id_t const &computePlaceId) {
		if (!_verboseThreadManagement) {
			return;
		}

		Instrument::InstrumentationContext tmpContext;
		tmpContext._threadId = threadId;
		tmpContext._computePlaceId = computePlaceId;

		LogEntry *logEntry = getLogEntry(tmpContext);
		assert(logEntry != nullptr);

		logEntry->appendLocation(tmpContext);
		logEntry->_contents << " <-> StartThread " << threadId;

		addLogEntry(logEntry);
	}

	void precreatedExternalThread(/* OUT */ external_thread_id_t &threadId) {
		threadId = GenericIds::getNewExternalThreadId();
	}

	void createdExternalThread_private(external_thread_id_t &threadId, std::string const &name) {
		if (!_verboseThreadManagement) {
			return;
		}

		InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent();

		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);

		logEntry->appendLocation(context);
		logEntry->_contents << " <-> CreateExternalThread " << name << " " << threadId;

		addLogEntry(logEntry);
	}

	void threadSynchronizationCompleted(
		__attribute((unused)) thread_id_t threadId
	) {
	}

	void threadWillSuspend(
		__attribute__((unused)) thread_id_t threadId,
		__attribute__((unused)) compute_place_id_t computePlaceID,
		__attribute__((unused)) bool afterSynchronization
	) {
		if (!_verboseThreadManagement) {
			return;
		}

		InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent();

		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);

		logEntry->appendLocation(context);
		logEntry->_contents << " --> SuspendThread ";

		addLogEntry(logEntry);
	}

	void threadSuspending(__attribute__((unused)) thread_id_t threadId)
	{
	}

	void threadBindRemote(__attribute__((unused)) thread_id_t threadId, __attribute__((unused)) compute_place_id_t cpu)
	{
	}

	void threadHasResumed(
		__attribute__((unused)) thread_id_t threadId,
		__attribute__((unused)) compute_place_id_t computePlaceID,
		__attribute__((unused)) bool afterSynchronization
	) {
		if (!_verboseThreadManagement) {
			return;
		}

		InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent();

		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);

		logEntry->appendLocation(context);
		logEntry->_contents << " <-- SuspendThread ";

		addLogEntry(logEntry);
	}

	void threadWillSuspend(__attribute__((unused)) external_thread_id_t threadId) {
		if (!_verboseThreadManagement) {
			return;
		}

		InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent();

		if (context._externalThreadName != nullptr) {
			if (!_verboseLeaderThread && (*context._externalThreadName == "leader-thread")) {
				return;
			}
		}

		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);

		logEntry->appendLocation(context);
		logEntry->_contents << " --> SuspendThread ";

		addLogEntry(logEntry);
	}


	void threadHasResumed(__attribute__((unused)) external_thread_id_t threadId) {
		if (!_verboseThreadManagement) {
			return;
		}

		InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent();

		if (context._externalThreadName != nullptr) {
			if (!_verboseLeaderThread && (*context._externalThreadName == "leader-thread")) {
				return;
			}
		}

		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);

		logEntry->appendLocation(context);
		logEntry->_contents << " <-- SuspendThread ";

		addLogEntry(logEntry);
	}

	static void verboseThreadWillShutdown() {
		if (!_verboseThreadManagement) {
			return;
		}

		InstrumentationContext const &context = ThreadInstrumentationContext::getCurrent();

		LogEntry *logEntry = getLogEntry(context);
		assert(logEntry != nullptr);

		logEntry->appendLocation(context);
		logEntry->_contents << " <-> ShutdownThread ";

		addLogEntry(logEntry);
	}

	void threadWillShutdown(__attribute__((unused)) external_thread_id_t threadId)
	{
		verboseThreadWillShutdown();
	}

	void threadWillShutdown()
	{
		verboseThreadWillShutdown();
	}
}
