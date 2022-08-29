/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2022 Barcelona Supercomputing Center (BSC)
*/

#ifndef INSTRUMENT_LEADER_THREAD_HPP
#define INSTRUMENT_LEADER_THREAD_HPP


namespace Instrument {
	void leaderThreadSpin();

	//! Called when the leader thread starts the body
	void leaderThreadBegin();

	//! Called when the leader thread ends the body
	void leaderThreadEnd();
}


#endif // INSTRUMENT_LEADER_THREAD_HPP
