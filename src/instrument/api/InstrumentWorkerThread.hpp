/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_WORKERTHREAD_HPP
#define INSTRUMENT_WORKERTHREAD_HPP

namespace Instrument {

	//! This function is called when the current worker thread loops on its main loop
	void workerThreadSpins();

	//! This function is called when the current worker thread obtains a task
	void workerThreadObtainedTasks();

	//! This function is called when the current worker thread starts busy waiting
	void workerThreadBusyWaits();

}

#endif // INSTRUMENT_WORKERTHREAD_HPP
