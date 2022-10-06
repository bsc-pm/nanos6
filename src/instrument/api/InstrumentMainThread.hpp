/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_MAINTHREAD_HPP
#define INSTRUMENT_MAINTHREAD_HPP

namespace Instrument {

	//! Called when the main thread begins
	void mainThreadBegin();

	//! Called when the main thread ends
	void mainThreadEnd();
}

#endif // INSTRUMENT_MAINTHREAD_HPP
