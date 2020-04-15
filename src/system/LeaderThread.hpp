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

	//! The LeaderThread's virtual CPU
	CPU *_leaderThreadCPU;

public:

	inline LeaderThread(CPU *leaderThreadCPU) :
		HelperThread("leader-thread"),
		_mustExit(false),
		_leaderThreadCPU(leaderThreadCPU)
	{
	}

	virtual ~LeaderThread()
	{
	}

	//! \brief Initialize the structures of the leader thread
	//!
	//! \param[in] leaderThreadCPU The leader thread's virtual CPU
	static void initialize(CPU *leaderThreadCPU);

	//! \brief Finalize the structures of the leader thread
	static void shutdown();

	//! \brief A loop that takes care of maintenance duties
	void body();

	//! \brief Check whether the leader thread is exiting
	//!
	//! \return true if leader thread is exiting
	static inline bool isExiting()
	{
		if (_singleton == nullptr) {
			return false;
		}

		return _singleton->_mustExit.load();
	}

	//! \brief Check whether the current thread is the leader thread
	//!
	//! \return true if the current thread is the leader
	static inline bool isLeaderThread()
	{
		KernelLevelThread *thread = static_cast<KernelLevelThread *> (getCurrentKernelLevelThread());
		if (thread == nullptr) {
			return false;
		}
		return (typeid(*thread) == typeid(LeaderThread));
	}

	//! \brief Get the virtual compute place of the leader thread
	//!
	//! \return The virtual compute place
	static inline CPU *getComputePlace()
	{
		return _singleton->_leaderThreadCPU;
	}
};

#endif // LEADER_THREAD_HPP
