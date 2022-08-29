/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_PTHREAD_HPP
#define INSTRUMENT_OVNI_PTHREAD_HPP

#include "instrument/api/InstrumentPthread.hpp"
#include "OvniTrace.hpp"

namespace Instrument {

	inline void pthreadBegin(pid_t creatorTid, int cpu, void *arg)
	{
		// Allocate the ovni buffer to hold events first
		Ovni::threadInit();

		uint64_t tag = (uint64_t) arg;
		Ovni::threadExecute(cpu, creatorTid, tag);
	}

	inline void pthreadEnd()
	{
		Ovni::threadEnd();

		// Flush events at the end of the thread life
		Ovni::flush();
	}

	inline void pthreadCreate(int cpu, void *arg)
	{
		uint64_t tag = (uint64_t) arg;
		Ovni::threadCreate(cpu, tag);
	}

	inline void pthreadBind(pid_t tid, int cpu)
	{
		Ovni::affinityRemote(cpu, tid);
	}

	inline void pthreadPause()
	{
		Ovni::threadPause();
	}

	inline void pthreadResume()
	{
		Ovni::threadResume();
	}

	inline void pthreadSignal(pid_t tid)
	{
		Ovni::threadSignal(tid);
	}

	inline void pthreadCool()
	{
		Ovni::threadCool();
	}
}

#endif // INSTRUMENT_OVNI_PTHREAD_HPP

