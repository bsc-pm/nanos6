#ifndef LEADER_THREAD_HPP
#define LEADER_THREAD_HPP


#include <atomic>

#include "lowlevel/KernelLevelThread.hpp"


//! \brief This class contains the code of the leader thread that consists in performing maintenance duties
class LeaderThread : public KernelLevelThread {
	static LeaderThread *_singleton;
	
	std::atomic<bool> _mustExit;
	
public:
	static void initialize();
	static void shutdown();
	
	LeaderThread();
	virtual ~LeaderThread()
	{
	}
	
	//! \brief A loop that takes care of maintenance duties
	//! The loop ends after notifyMainExit is called
	void *body();
	
	static bool isExiting()
	{
		if (_singleton == nullptr) {
			return false;
		}
		
		return _singleton->_mustExit.load();
	}
};


#endif // LEADER_THREAD_HPP
