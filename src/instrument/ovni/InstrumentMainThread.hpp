/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_OVNI_MAINTHREAD_HPP
#define INSTRUMENT_OVNI_MAINTHREAD_HPP

#include "instrument/api/InstrumentMainThread.hpp"
#include "OvniTrace.hpp"

namespace Instrument {

	inline void mainThreadBegin()
	{
		// We use the mainThreadBegin to initialize ovni, as is the
		// earliest point we can instrument

		// Check the ovni version before calling any other ovni function
		Ovni::checkVersion();

		Ovni::procInit();
		Ovni::threadInit();
		Ovni::threadExecute(-1, -1, 0);
		Ovni::genBursts();
		Ovni::threadTypeBegin('M');
	}

	inline void mainThreadEnd()
	{
		// Similarly, we use the mainThreadEnd as the last possible
		// point where we stop the thread and process
		Ovni::threadTypeEnd('M');
		Ovni::threadEnd();
		Ovni::procFini();
	}

}

#endif // INSTRUMENT_OVNI_MAINTHREAD_HPP

