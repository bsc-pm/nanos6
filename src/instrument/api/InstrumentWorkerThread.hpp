/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020-2023 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_WORKERTHREAD_HPP
#define INSTRUMENT_WORKERTHREAD_HPP

namespace Instrument {

	//! This function is called when the current worker thread loops on its main loop
	void workerThreadSpins();

	//! This function is called when the current worker thread obtains a task
	void workerThreadObtainedTasks();

	//! Called when the thread enters or exits the idle state.
	void workerIdle(bool isIdle);

	//! This function is called when the current worker thread starts busy waiting
	void workerThreadBusyWaits();

	//! Called when the current worker thread starts the body
	void workerThreadBegin();

	//! Called when the current worker thread ends the body
	void workerThreadEnd();

	//! Called when the current worker enters handleTask()
	void enterHandleTask();

	//! Called when the current worker exits handleTask()
	void exitHandleTask();

	//! Called when the current worker enters switchTo()
	void enterSwitchTo();

	//! Called when the current worker exits switchTo()
	void exitSwitchTo();

	//! Called when the current worker enters suspend()
	void enterSuspend();

	//! Called when the current worker exits suspend()
	void exitSuspend();

	//! Called when the current worker enters resume()
	void enterResume();

	//! Called when the current worker exits resume()
	void exitResume();
}

#endif // INSTRUMENT_WORKERTHREAD_HPP
