/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_WORKERTHREAD_HPP
#define INSTRUMENT_WORKERTHREAD_HPP

namespace Instrument {
	void WorkerSpins();
	void WorkerGetsTasks();
	void WorkerBusyWaits();
}

#endif // INSTRUMENT_WORKERTHREAD_HPP
