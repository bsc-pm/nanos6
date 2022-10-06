/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_WORKERTHREAD_HPP
#define INSTRUMENT_OVNI_WORKERTHREAD_HPP

#include "instrument/api/InstrumentWorkerThread.hpp"
#include "OvniTrace.hpp"

namespace Instrument {

	inline void workerThreadSpins() {}
	inline void workerThreadObtainedTask() {}
	inline void workerThreadBusyWaits() {}

	inline void workerThreadBegin()
	{
		Ovni::threadTypeBegin('W');
		Ovni::workerLoopEnter();
	}

	inline void workerThreadEnd()
	{
		Ovni::workerLoopExit();
		Ovni::threadTypeEnd('W');
	}

	inline void enterHandleTask()
	{
		Ovni::handleTaskEnter();
	}

	inline void exitHandleTask()
	{
		Ovni::handleTaskExit();
	}

	inline void enterSwitchTo()
	{
		Ovni::switchToEnter();
	}

	inline void exitSwitchTo()
	{
		Ovni::switchToExit();
	}

	inline void enterSuspend()
	{
		Ovni::suspendEnter();
	}

	inline void exitSuspend()
	{
		Ovni::suspendExit();
	}

	inline void enterResume()
	{
		Ovni::resumeEnter();
	}

	inline void exitResume()
	{
		Ovni::resumeExit();
	}
}

#endif // INSTRUMENT_OVNI_WORKERTHREAD_HPP

