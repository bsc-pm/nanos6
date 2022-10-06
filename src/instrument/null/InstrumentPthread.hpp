/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_NULL_PTHREAD_HPP
#define INSTRUMENT_NULL_PTHREAD_HPP

#include "instrument/api/InstrumentPthread.hpp"

namespace Instrument {

	inline void pthreadBegin(
			__attribute__((unused)) pid_t creatorTid,
			__attribute__((unused)) int cpu,
			__attribute__((unused)) void *arg) {}
	inline void pthreadEnd() {}
	inline void pthreadCreate(
			__attribute__((unused)) int cpu,
			__attribute__((unused)) void *arg) {}
	inline void pthreadBind(
			__attribute__((unused)) pid_t tid,
			__attribute__((unused)) int cpu) {}
	inline void pthreadPause() {}
	inline void pthreadResume() {}
	inline void pthreadCool() {}
	inline void pthreadSignal(__attribute__((unused)) pid_t tid) {}
}

#endif // INSTRUMENT_NULL_PTHREAD_HPP
