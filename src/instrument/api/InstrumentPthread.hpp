/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_PTHREAD_HPP
#define INSTRUMENT_PTHREAD_HPP

#include <unistd.h> // For pid_t

namespace Instrument {
	// Called from a newly created pthread
	void pthreadBegin(pid_t creatorTid, int cpu, void *arg);

	// Called when a pthread ends the `start_routine`
	void pthreadEnd();

	// Called when this thread creates another with affinity in the
	// given CPU. The arg pointer must match the one in the pthreadBegin
	// from the child.
	void pthreadCreate(int cpu, void *arg);

	// Called when the pthread (tid) sets the affinity to the given cpu.
	// Use cpu = -1 if the affinity mask has more than 1 bit set.
	void pthreadBind(pid_t tid, int cpu);

	// Called before the pthread enters the condition variable wait
	void pthreadPause();

	// Called after the pthread exits the condition variable wait
	void pthreadResume();

	// Called when other threads can begin the execution before this one
	// ends or pauses.
	void pthreadCool();

	// Called when waking another pthread with the given tid
	void pthreadSignal(pid_t tid);
}

#endif // INSTRUMENT_PTHREAD_HPP
