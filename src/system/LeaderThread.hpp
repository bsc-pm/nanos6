/*
	This file is part of Nanos6 and is licensed under the terms contained in the COPYING file.

	Copyright (C) 2015-2020 Barcelona Supercomputing Center (BSC)
*/

#ifndef LEADER_THREAD_HPP
#define LEADER_THREAD_HPP

#include <atomic>

#include "executors/threads/CPU.hpp"
#include "lowlevel/threads/HelperThread.hpp"


//! \brief This class contains the code of the leader threat, which
//! performs maintenance duties
class LeaderThread : public HelperThread {

private:

	//! The singleton instance
	static LeaderThread *_singleton;

	//! Whether the LeaderThread must stop executing
	std::atomic<bool> _mustExit;

	//! The LeaderThread's virtual CPU (for instrumentation purposes)
	static CPU *_leaderThreadCPU;

public:

	inline LeaderThread() :
		HelperThread("leader-thread"),
		_mustExit(false)
	{
	}

	virtual ~LeaderThread()
	{
	}

	static void initialize();

	static void shutdown();

	//! \brief A loop that takes care of maintenance duties
	void body();

	static inline bool isExiting()
	{
		if (_singleton == nullptr) {
			return false;
		}

		return _singleton->_mustExit.load();
	}

	static inline void setCPU(CPU *cpu)
	{
		_leaderThreadCPU = cpu;
	}

	static inline CPU *getCPU()
	{
		return _leaderThreadCPU;
	}
};

#endif // LEADER_THREAD_HPP
